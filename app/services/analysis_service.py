import os
import boto3
import torch
import tempfile   
import subprocess  
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.schemas.analysis_schema import AICodeDetectionRequest, AICodeDetectionResponse

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

# 모델을 평가(추론) 모드로 전환합니다. (드롭아웃 같은 학습용 기능 비활성화)
model.eval()

print("완료")

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


async def detect_ai_code(request: AICodeDetectionRequest) -> AICodeDetectionResponse:
    # 스프링 부트(또는 포스트맨)에서 전달받은 원본 코드 내용
    code = request.code_content
    
    security_report = run_semgrep(code)

    # max_length=512: 모델이 한 번에 읽을 수 있는 최대 길이로 커팅
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    
    # 모델 추론 시작
    with torch.no_grad(): 
        outputs = model(**inputs)
        
        # 로짓 값을 0.0 ~ 1.0 사이의 긍정/부정 확률로 변환
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # index 0 = 사람, index 1 = AI
        # 작성 확률을 퍼센트 단위로 변환
        ai_prob = probabilities[0][1].item() * 100.0 
        
        # 확률 > 50% = AI가 짠 코드
        is_ai = ai_prob >= 50.0

    # 결과 반환 (DTO에 맞춰서 포장)
    return AICodeDetectionResponse(
        is_ai_generated=is_ai,
        ai_probability=round(ai_prob, 2),
        has_vulnerability=security_report["has_vulnerability"],
        vulnerabilities=security_report["vulnerabilities"]
    )