import os
import boto3
import torch
import tempfile   
import subprocess  
import json
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.schemas.analysis_schema import AICodeDetectionRequest, AICodeDetectionResponse
from openai import OpenAI
import lizard
import re

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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" 
)
# 1. 모델이 저장된 폴더 경로
MODEL_PATH = "./app/models/codebart"
MODEL_FILE = os.path.join(MODEL_PATH, "model.safetensors") 

S3_BUCKET = "guardrail-codebert-models-v1"
S3_KEY = "codebart/model.safetensors"
AWS_REGION = "ap-southeast-1"

def download_model_if_needed():
    """로컬에 모델 파일이 없으면 S3에서 다운로드"""
    if os.path.exists(MODEL_FILE):
        print("모델이 이미 로컬에 존재합니다. 다운로드 스킵.")
        return

    print(f"모델 다운로드 중... (S3: {S3_BUCKET}/{S3_KEY})")
    os.makedirs(MODEL_PATH, exist_ok=True)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_KEY, MODEL_FILE)
    print("모델 다운로드 완료.")

print("모델 업로드중")

download_model_if_needed()
# 모델 추론용 자료로 서버 처음 켜질 때 한 번 로드 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("완료")

def generate_patched_code(original_code: str, vulnerabilities: list, needs_refactoring: bool = False, max_complexity: int = 0) -> str:
    
    refactoring_instruction = ""
    if needs_refactoring:
        refactoring_instruction = f"\n- [알고리즘 최적화]: 이 코드는 순환 복잡도가 {max_complexity}로 매우 높습니다. 불필요한 중첩 루프와 조건문을 제거하여 시간 복잡도를 줄이고 클린 코드로 리팩토링하세요."

    ticks = "`" * 3

    system_prompt = f"""
    당신은 세계 최고의 보안 코딩 및 알고리즘 최적화 전문가입니다.
    사용자가 코드를 주면, 보안 취약점을 해결하고 리팩토링한 완성본 코드를 제공해야 합니다.
    [절대 규칙]
    - 로직의 원래 의미(비즈니스 로직)는 절대 변경하지 말고 구조만 개선하세요.
    - 코드는 반드시 마크다운 코드 블록({ticks}java 와 {ticks}) 안에 작성하세요.
    - 코드 블록 밖에는 어떠한 설명도 적지 마세요.
    """

    user_prompt = f"""
    [발견된 보안 취약점 리스트]
    {json.dumps(vulnerabilities, ensure_ascii=False, indent=2)}
    
    [수정 요청 사항]
    - 발견된 보안 취약점(SQL Injection 등)을 완벽하게 패치하세요.{refactoring_instruction}
    
    [원본 코드]
    {original_code}
    """

    try:
        print("LLaMA 보완코드 생성 시작")
        
        completion = client.chat.completions.create(
          model="meta/llama-3.1-8b-instruct",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
          ], 
          temperature=0.1,
          top_p=0.1,
          max_tokens=2048,
          stream=False
        )
        
        raw_output = completion.choices[0].message.content.strip()
        
        # 생성한 변수(ticks)를 이용해 정규식 패턴을 조립합니다.
        pattern = ticks + r'(?:java|c|python)?\n(.*?)\n' + ticks
        match = re.search(pattern, raw_output, re.DOTALL)
        
        if match:
            final_code = match.group(1).strip()
        else:
            final_code = raw_output.replace(ticks + "java", "").replace(ticks, "").strip()
            
        return final_code
        
    except Exception as e:
        print(f"LLaMA API 호출 중 에러 발생: {e}")
        return "보완 코드 생성에 실패했습니다."

def run_semgrep(code_content: str) -> dict:
    """코드를 임시 파일로 만들어 Semgrep으로 보안 취약점을 검사하는 함수"""
    
    # 1. 코드를 담을 임시 자바 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(code_content)
        temp_file_path = temp_file.name

    try:
        # 2. Semgrep 실행
        result = subprocess.run(
            ['semgrep', '--config', 'p/java', '--json', temp_file_path],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # 3. 결과 파싱
        output_data = json.loads(result.stdout)
        results = output_data.get('results', [])
        
        # 4. 프론트엔드/스프링 전송용 데이터 정리
        vulnerabilities = []
        for item in results:
            vulnerabilities.append({
                "rule_id": item['check_id'],
                "message": item['extra']['message'],
                "line": item['start']['line']
            })
            
        return {
            "has_vulnerability": len(vulnerabilities) > 0,
            "vulnerabilities": vulnerabilities
        }
        
    except Exception as e:
        print(f"Semgrep 실행 중 에러 발생: {e}")
        return {"has_vulnerability": False, "vulnerabilities": []}
        
    finally:
        # 5. 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def analyze_complexity(code_content: str) -> dict:
    """Lizard를 사용해 함수의 복잡도를 분석하고, 기준치를 초과한 함수의 상세 정보를 반환합니다."""
    try:
        analysis = lizard.analyze_file.analyze_source_code("temp.java", code_content)
        
        if not analysis.function_list:
            return {"max_complexity": 0, "details": []}
            
        details = []
        max_comp = 0
        COMPLEXITY_THRESHOLD = 15
        
        for func in analysis.function_list:
            comp = func.cyclomatic_complexity
            max_comp = max(max_comp, comp)
            
            if comp > COMPLEXITY_THRESHOLD:
                details.append({
                    "function_name": func.name,
                    "complexity_score": comp,
                    "line": func.start_line,
                    "message": f"'{func.name}' 함수의 순환 복잡도가 {comp}로 매우 높습니다. 불필요한 중첩 루프(for/while)나 조건문(if)을 분리하여 O(N) 단위로 최적화가 필요합니다."
                })
                
        return {
            "max_complexity": max_comp,
            "details": details
        }
    except Exception as e:
        print(f"Lizard 분석 중 에러 발생: {e}")
        return {"max_complexity": 0, "details": []}


def run_security_pipeline(code: str) -> dict:
    
    security_report = run_semgrep(code)
    complexity_report = analyze_complexity(code)
    
    max_complexity = complexity_report["max_complexity"]
    complexity_details = complexity_report["details"]
    
    COMPLEXITY_THRESHOLD = 15
    needs_refactoring = max_complexity > COMPLEXITY_THRESHOLD
    
    patched_code_result = None
    
    if security_report["has_vulnerability"] or needs_refactoring:
        patched_code_result = generate_patched_code(
            code, 
            security_report["vulnerabilities"], 
            needs_refactoring, 
            max_complexity
        )
        
    return {
        "has_vulnerability": security_report["has_vulnerability"],
        "vulnerabilities": security_report["vulnerabilities"],
        "max_complexity": max_complexity,          
        "needs_refactoring": needs_refactoring,
        "complexity_details": complexity_details,  
        "patched_code": patched_code_result
    }

def run_ai_detection(code: str) -> dict:
    """AI 모델을 돌려 확률을 계산하는 묶음 함수"""
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        ai_prob = probabilities[0][1].item() * 100.0
        is_ai = ai_prob >= 50.0
        
    return {
        "is_ai_generated": is_ai,
        "ai_probability": round(ai_prob, 2)
    }

async def detect_ai_code(request: AICodeDetectionRequest) -> AICodeDetectionResponse:
    code = request.code_content

    security_task = asyncio.to_thread(run_security_pipeline, code)
    ai_task = asyncio.to_thread(run_ai_detection, code)
    
    security_result, ai_result = await asyncio.gather(security_task, ai_task)
    
    return AICodeDetectionResponse(
        is_ai_generated=ai_result["is_ai_generated"],
        ai_probability=ai_result["ai_probability"],
        has_vulnerability=security_result["has_vulnerability"],
        vulnerabilities=security_result["vulnerabilities"],
        max_complexity=security_result["max_complexity"],
        needs_refactoring=security_result["needs_refactoring"],
        complexity_details=security_result.get("complexity_details", []),  
        patched_code=security_result["patched_code"]
    )