import os
import boto3
import torch
import tempfile   
import subprocess  
import json
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.schemas.analysis_schema import AICodeDetectionRequest, AICodeDetectionResponse, PromptReconstructRequest, PromptReconstructResponse
from openai import OpenAI
import lizard
import re

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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" 
)
MODEL_PATH = "./app/models/codebart"
MODEL_FILE = os.path.join(MODEL_PATH, "model.safetensors") 

S3_BUCKET = "guardrail-codebert-models-v1"
S3_KEY = "codebart/model.safetensors"
AWS_REGION = "ap-southeast-1"

LANGUAGE_CONFIG = {
    "java": {"semgrep_configs": ["p/java"], "extension": ".java"},
    "python": {"semgrep_configs": ["p/python", "./rules/custom_python_sqli.yaml"], "extension": ".py"},
    "javascript": {"semgrep_configs": ["p/javascript"], "extension": ".js"},
    "typescript": {"semgrep_configs": ["p/typescript"], "extension": ".ts"},
}

LANGUAGE_ALIASES = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "java": "java",
}

DEFAULT_LANGUAGE = "java"


def get_language_config(language: str) -> dict:
    key = (language or DEFAULT_LANGUAGE).lower()
    key = LANGUAGE_ALIASES.get(key, key)
    return LANGUAGE_CONFIG.get(key, LANGUAGE_CONFIG[DEFAULT_LANGUAGE])


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
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("완료")


def translate_vulnerabilities_to_korean(vulnerabilities: list) -> list:
    """Semgrep 취약점 메시지만 한 번에 모아서 한국어로 번역 (Lizard는 이미 한국어라 대상 아님)"""
    if not vulnerabilities:
        return vulnerabilities

    joined = "\n---\n".join(v["message"] for v in vulnerabilities)

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": (
                    "당신은 보안 취약점 설명 번역가입니다. "
                    "입력된 여러 문장은 '---'로 구분되어 있습니다. "
                    "각 문장을 자연스러운 한국어로 번역하세요. "
                    "코드 식별자, 함수명, 라이브러리명(예: MD5, SHA256, subprocess)은 번역하지 말고 그대로 두세요. "
                    "번역 결과만 입력과 동일한 개수로, 동일하게 '---'로 구분해서 출력하세요. 다른 설명은 붙이지 마세요."
                )},
                {"role": "user", "content": joined}
            ],
            temperature=0.1,
            top_p=0.1,
            max_tokens=1024,
            stream=False
        )
        translated_joined = completion.choices[0].message.content.strip()
        translated_list = [t.strip() for t in translated_joined.split("---")]

        if len(translated_list) == len(vulnerabilities):
            for v, translated in zip(vulnerabilities, translated_list):
                v["message"] = translated
        else:
            print("번역 결과 개수 불일치, 원문 유지")
    except Exception as e:
        print(f"취약점 메시지 번역 중 에러 발생: {e}")

    return vulnerabilities


def generate_patched_code(original_code: str, vulnerabilities: list, needs_refactoring: bool = False, 
                           max_complexity: int = 0, language: str = DEFAULT_LANGUAGE,
                           duplicate_snippets: list = None) -> str:
    
    refactoring_instruction = ""
    if needs_refactoring:
        refactoring_instruction = f"\n- [알고리즘 최적화]: 이 코드는 순환 복잡도가 {max_complexity}로 매우 높습니다. 불필요한 중첩 루프와 조건문을 제거하여 시간 복잡도를 줄이고 클린 코드로 리팩토링하세요."

    duplicate_instruction = ""
    if duplicate_snippets:
        snippets_text = "\n\n".join(
            f"[유사도 {d.get('similarity_score', 0):.2f}, 위치: {d.get('file_path')} "
            f"({d.get('start_line')}~{d.get('end_line')}줄), 함수명: {d.get('function_name')}, "
            f"매개변수: {', '.join(d.get('parameters') or [])}]\n{d.get('code')}"
            for d in duplicate_snippets[:2]
        )
        duplicate_instruction = f"""
    - [코드 재사용]: 아래는 이 프로젝트에 이미 존재하는 유사한 함수입니다.
      가능하다면 원본 코드에서 중복되는 함수 정의를 제거하고, 아래 기존 함수를 import(또는 참조)하여
      호출하는 방식으로 리팩토링하세요. 함수명과 매개변수 순서를 정확히 맞추세요.
      기존 로직의 실제 동작 방식은 절대 변경하지 마세요.

      {snippets_text}
    """

    ticks = "`" * 3
    lang_tag = (language or DEFAULT_LANGUAGE).lower()

    system_prompt = f"""
    당신은 세계 최고의 보안 코딩 및 알고리즘 최적화 전문가입니다.
    사용자가 코드를 주면, 보안 취약점을 해결하고 리팩토링한 완성본 코드를 제공해야 합니다.
    [절대 규칙]
    - 로직의 원래 의미(비즈니스 로직)는 절대 변경하지 말고 구조만 개선하세요.
    - 코드는 반드시 마크다운 코드 블록({ticks}{lang_tag} 와 {ticks}) 안에 작성하세요.
    - 코드 블록 밖에는 어떠한 설명도 적지 마세요.
    """

    user_prompt = f"""
    [발견된 보안 취약점 리스트]
    {json.dumps(vulnerabilities, ensure_ascii=False, indent=2)}
    
    [수정 요청 사항]
    - 발견된 보안 취약점(SQL Injection 등)을 완벽하게 패치하세요.{refactoring_instruction}{duplicate_instruction}
    
    [원본 코드]
    {original_code}
    """

    try:
        print("LLaMA 보완코드 생성 시작")
        estimated_code_tokens = max(1, len(original_code) // 3)
        expected_output_tokens = int(estimated_code_tokens * 1.3) + 512

        sample_max_tokens =  max(2048, min(expected_output_tokens, 8192))
        print(len(original_code))
        print("나눈 값: ",len(original_code) // 1024)
        print("현제 max 토큰:",sample_max_tokens)
        completion = client.chat.completions.create(
          model="meta/llama-3.1-8b-instruct",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
          ], 
          temperature=0.1,
          top_p=0.1,
          max_tokens = sample_max_tokens,
          stream=False
        )
        raw_output = completion.choices[0].message.content.strip()
        pattern = ticks + r'(?:\w+)?\n(.*?)\n' + ticks
        match = re.search(pattern, raw_output, re.DOTALL)

        if match:
            final_code = match.group(1).strip()
        else:
            final_code = raw_output.replace(ticks + lang_tag, "").replace(ticks, "").strip()

        return final_code

    except Exception as e:
        print(f"LLaMA API 호출 중 에러 발생: {e}")
        return "보완 코드 생성에 실패했습니다."


def reconstruct_prompt(original_prompt: str, code: str, vulnerabilities: list, complexity_details: list) -> dict:
    """사용자의 원본 프롬프트를, 발견된 문제점이 재발하지 않도록 재구성"""

    issues_text = ""
    if vulnerabilities:
        vuln_list = "\n".join(f"- {v.get('message', '')}" for v in vulnerabilities[:5])
        issues_text += f"\n[발견된 보안 취약점]\n{vuln_list}\n"
    if complexity_details:
        comp_list = "\n".join(f"- {c.get('message', '')}" for c in complexity_details[:5])
        issues_text += f"\n[발견된 복잡도/비효율 이슈]\n{comp_list}\n"

    if not issues_text:
        issues_text = "\n(특별히 발견된 취약점이나 비효율 이슈는 없었습니다. 다만 일반적인 코드 품질 관점에서 프롬프트를 다듬어주세요.)\n"

    system_prompt = """
당신은 프롬프트 엔지니어링 전문가입니다.
사용자가 AI에게 코드 생성을 요청했던 "원본 프롬프트"와, 그 결과로 생성된 코드에서 발견된 문제점을 받게 됩니다.
같은 목적(기능)을 달성하되, 발견된 문제가 재발하지 않도록 프롬프트를 재작성해야 합니다.

[절대 규칙]
- 원본 프롬프트의 핵심 목적/기능 요구사항은 절대 바꾸지 마세요.
- "안전하게", "적절히", "올바르게", "효율적으로" 같은 추상적이고 모호한 표현은 절대 사용하지 마세요.
- 반드시 구체적인 기술 용어/기법명을 직접 명시하세요.
- **입력된 문제점 목록(취약점 + 복잡도) 각각에 대해, 빠짐없이 하나씩 프롬프트 본문에 구체적인 지시 문장을 추가하세요.
  일부만 반영하고 나머지를 설명(explanation)에서만 언급하는 것은 금지합니다.**
- **비밀값/키를 환경변수 등으로 옮기라고 지시할 때는, "값이 없을 경우 기본값을 하드코딩하지 말고 
  에러를 발생시키거나 실행을 중단하라"는 지시도 함께 포함하세요.**
- 반드시 다음 형식으로만 응답하세요:
[PROMPT]
(재구성된 프롬프트 전체)
[EXPLANATION]
(어떤 부분을 왜 추가/수정했는지 2~3문장으로 설명)
"""

    user_prompt = f"""
    [원본 프롬프트]
    {original_prompt}

    [그 프롬프트로 생성된 코드]
    {code}

    [이 코드에서 발견된 문제점]
    {issues_text}

    위 문제가 재발하지 않도록, 원본 프롬프트를 재구성해주세요.
    """

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            top_p=0.2,
            max_tokens=2048,
            stream=False
        )
        raw_output = completion.choices[0].message.content.strip()

        prompt_match = re.search(r'\[PROMPT\]\s*(.*?)\s*\[EXPLANATION\]', raw_output, re.DOTALL)
        explanation_match = re.search(r'\[EXPLANATION\]\s*(.*)', raw_output, re.DOTALL)

        reconstructed_prompt = prompt_match.group(1).strip() if prompt_match else raw_output
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        return {
            "reconstructed_prompt": reconstructed_prompt,
            "explanation": explanation
        }

    except Exception as e:
        print(f"프롬프트 재구성 중 에러 발생: {e}")
        return {
            "reconstructed_prompt": original_prompt,
            "explanation": "프롬프트 재구성에 실패하여 원본 프롬프트를 그대로 반환합니다."
        }


async def reconstruct_prompt_endpoint(request: PromptReconstructRequest) -> PromptReconstructResponse:
    result = reconstruct_prompt(
        request.original_prompt,
        request.code_content,
        request.vulnerabilities,
        request.complexity_details
    )
    return PromptReconstructResponse(
        reconstructed_prompt=result["reconstructed_prompt"],
        explanation=result["explanation"]
    )


def run_semgrep(code_content: str, language: str = DEFAULT_LANGUAGE) -> dict:
    """코드를 임시 파일로 만들어 Semgrep으로 보안 취약점을 검사하는 함수"""
    
    lang_cfg = get_language_config(language)

    with tempfile.NamedTemporaryFile(mode='w', suffix=lang_cfg["extension"], delete=False, encoding='utf-8') as temp_file:
        temp_file.write(code_content)
        temp_file_path = temp_file.name

    try:
        cmd = ['semgrep']
        for config in lang_cfg["semgrep_configs"]:
            cmd += ['--config', config]
        cmd += ['--json', temp_file_path]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        output_data = json.loads(result.stdout)
        results = output_data.get('results', [])
        
        vulnerabilities = []
        for item in results:
            vulnerabilities.append({
                "rule_id": item['check_id'],
                "message": item['extra']['message'],
                "line": item['start']['line']
            })

        vulnerabilities = translate_vulnerabilities_to_korean(vulnerabilities)
        return {
            "has_vulnerability": len(vulnerabilities) > 0,
            "vulnerabilities": vulnerabilities
        }
        
    except Exception as e:
        print(f"Semgrep 실행 중 에러 발생: {e}")
        return {"has_vulnerability": False, "vulnerabilities": []}
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def analyze_complexity(code_content: str, language: str = DEFAULT_LANGUAGE) -> dict:
    """Lizard를 사용해 함수의 복잡도를 분석하고, 기준치를 초과한 함수의 상세 정보를 반환합니다."""
    lang_cfg = get_language_config(language)
    filename = "temp" + lang_cfg["extension"]

    try:
        analysis = lizard.analyze_file.analyze_source_code(filename, code_content)
        
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


def run_security_pipeline(code: str, language: str = DEFAULT_LANGUAGE, duplicate_snippets: list = None) -> dict:
    
    security_report = run_semgrep(code, language)
    complexity_report = analyze_complexity(code, language)
    
    max_complexity = complexity_report["max_complexity"]
    complexity_details = complexity_report["details"]
    
    COMPLEXITY_THRESHOLD = 15
    needs_refactoring = max_complexity > COMPLEXITY_THRESHOLD
    
    patched_code_result = None
    
    if security_report["has_vulnerability"] or needs_refactoring or duplicate_snippets:
        patched_code_result = generate_patched_code(
            code, 
            security_report["vulnerabilities"], 
            needs_refactoring, 
            max_complexity,
            language,
            duplicate_snippets
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
    language = getattr(request, "language", None) or DEFAULT_LANGUAGE

    print(f"[DEBUG] 받은 duplicate_snippets: {request.duplicate_snippets}")

    # Spring이 이미 검색/보강해서 넘겨준 값을 그대로 사용 (FastAPI가 직접 검색하지 않음)
    duplicate_snippets = [d.model_dump() for d in (request.duplicate_snippets or [])]

    print(f"[DEBUG] 변환된 duplicate_snippets: {duplicate_snippets}")

    security_task = asyncio.to_thread(run_security_pipeline, code, language, duplicate_snippets)
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