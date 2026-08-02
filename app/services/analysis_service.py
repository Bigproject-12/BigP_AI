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
  api_key = "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",
)
MODEL_PATH = "./app/models/codebart"
MODEL_FILE = os.path.join(MODEL_PATH, "model.safetensors") 

S3_BUCKET = "guardrail-codebert-models-v1"
S3_KEY = "codebart/model.safetensors"
AWS_REGION = "ap-southeast-1"

LANGUAGE_CONFIG = {
    "java": {"semgrep_configs": ["p/java", "./rules/custom_java_rules.yaml"], "extension": ".java"},
    "python": {"semgrep_configs": ["p/python", "./rules/custom_python_rules.yaml"], "extension": ".py"},
    "javascript": {"semgrep_configs": ["p/javascript"], "extension": ".js"},
    "typescript": {"semgrep_configs": ["p/typescript"], "extension": ".ts"},
    "c": {"semgrep_configs": ["p/c", "./rules/custom_c_rules.yaml"], "extension": ".c"},
    "cpp": {"semgrep_configs": ["p/cpp", "./rules/custom_cpp_rules.yaml"], "extension": ".cpp"},
    "csharp": {"semgrep_configs": ["p/csharp", "./rules/custom_csharp_rules.yaml"], "extension": ".cs"},
}

LANGUAGE_ALIASES = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cs": "csharp",
    "csharp": "csharp",
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
            model="google/diffusiongemma-26b-a4b-it",
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

    lang_tag = (language or DEFAULT_LANGUAGE).lower()

    system_prompt = f"""
    당신은 세계 최고의 보안 코딩 및 알고리즘 최적화 전문가입니다.
    사용자가 코드를 주면, 보안 취약점을 해결하고 리팩토링한 완성본 코드를 제공해야 합니다.
    [절대 규칙]
    - 로직의 원래 의미(비즈니스 로직)는 절대 변경하지 말고 구조만 개선하세요.
    - 절대 마크다운 헤더(#, ##), 설명 문단, 목록(1. 2. 3.) 등을 포함하지 마세요.
    - 절대 마크다운 코드 블록(백틱 3개)을 사용하지 마세요.
    - 반드시 아래와 같은 순수 JSON 형식으로만 응답하세요. JSON 앞뒤에 어떤 텍스트도 붙이지 마세요:
    {{"patched_code": "여기에 완성된 {lang_tag} 코드 전체를 하나의 문자열로 작성 (줄바꿈은 \\n으로 표현)"}}
    """

    user_prompt = f"""
    [발견된 보안 취약점 리스트]
    {json.dumps(vulnerabilities, ensure_ascii=False, indent=2)}
    
    [수정 요청 사항]
    - 발견된 보안 취약점(SQL Injection 등)을 완벽하게 패치하세요.{refactoring_instruction}{duplicate_instruction}
    
    [원본 코드]
    {original_code}

    반드시 JSON 형식으로만 응답하세요.
    """

    try:
        print("DiffusionGemma 보완코드 생성 시작")
        estimated_code_tokens = max(1, len(original_code) // 3)
        expected_output_tokens = int(estimated_code_tokens * 1.3) + 512
        sample_max_tokens = max(2048, min(expected_output_tokens, 8192))

        completion = client.chat.completions.create(
          model="google/diffusiongemma-26b-a4b-it",
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
          ], 
          temperature=0.1,
          top_p=0.1,
          max_tokens=sample_max_tokens,
          stream=False
        )
        raw_output = completion.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw_output, strict=False)
            return parsed.get("patched_code", raw_output)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0), strict=False)
                return parsed.get("patched_code", raw_output)
            except json.JSONDecodeError:
                pass

        ticks = "`" * 3
        pattern = ticks + r'(?:\w+)?\n(.*?)\n' + ticks
        code_match = re.search(pattern, raw_output, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        loose_match = re.search(r'"patched_code"\s*:\s*"(.*)"\s*\}?\s*$', raw_output, re.DOTALL) # 이거까지 했는데 안 되면 죽임ㅇㅇ
        if loose_match:
            extracted = loose_match.group(1)
            extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            return extracted.strip()

        return raw_output

    except Exception as e:
        print(f"DiffusionGemma API 호출 중 에러 발생: {e}")
        return "보완 코드 생성에 실패했습니다."

def summarize_function(function_info: dict) -> str:
    """재사용 후보 함수 하나를, 다른 AI가 참고할 수 있는 자연어 설명으로 요약"""
    
    function_name = function_info.get("function_name", "")
    print(f"[LOG] summarize_function 시작: {function_name}")
    file_path = function_info.get("file_path", "")
    parameters = function_info.get("parameters") or []
    code = function_info.get("code", "")

    system_prompt = """
    당신은 코드 문서화 전문가입니다.
    주어진 함수의 원문 코드를 분석해서, 이 함수를 한 번도 본 적 없는 다른 개발자가 
    코드를 직접 보지 않고도 정확히 이해하고 사용할 수 있도록 자연어로 설명해야 합니다.
    
    [절대 규칙]
    - 반드시 다음 내용을 포함하세요: 
      1) 어떤 매개변수를 받는지(타입과 의미)
      2) 무엇을 반환하는지(타입과 그 안에 어떤 정보가 담기는지)
      3) 내부적으로 정확히 어떤 처리를 하는지
    - 코드를 그대로 복사하지 말고, 완전히 자연어 문장으로 설명하세요.
    - "안전하게", "적절히" 같은 모호한 표현 없이, 구체적으로 설명하세요.
    - 반드시 2~4문장의 순수 텍스트로만 응답하세요. 다른 설명, 마크다운, JSON을 포함하지 마세요.
    """

    user_prompt = f"""
    [함수명] {function_name}
    [매개변수] {', '.join(parameters)}
    [원문 코드]
    {code}

    위 함수를 자연어로 요약해주세요.
    """

    try:
        completion = client.chat.completions.create(
            model="google/diffusiongemma-26b-a4b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.1,
            max_tokens=512,
            stream=False
        )
        print(f"[LOG] summarize_function 완료: {function_name}")
        summary = completion.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"함수 요약 중 에러 발생: {e}")
        # 실패 시 최소한의 정보라도 반환
        return f"{function_name}({', '.join(parameters)}) 함수 (요약 생성 실패, 매개변수만 참고)"


def draft_reconstruct_prompt(original_prompt: str, code: str, issues_text: str, feedback: str = None) -> dict:
    """프롬프트 초안 작성. feedback이 있으면 이전 시도의 문제점을 참고해서 다시 작성"""
    print("[LOG] draft_reconstruct_prompt 시작")

    feedback_instruction = ""
    if feedback:
        feedback_instruction = f"""
        [이전 시도에서 발견된 문제점 - 반드시 이번엔 고쳐서 작성하세요]
        {feedback}
        """

    system_prompt = """
    당신은 프롬프트 엔지니어링 전문가입니다.
    원본 프롬프트와 발견된 문제점을 받아, 문제가 재발하지 않도록 프롬프트를 재작성하세요.
    이 재구성된 프롬프트는 이 프로젝트 코드에 전혀 접근할 수 없는 다른 AI 도구에게 그대로 전달됩니다.

    [규칙]
    - "재사용 가능한 기존 함수" 항목엔 이미 완성된 "기능 설명"이 주어집니다. 
        그 설명을 자연스러운 완결된 문장으로 프롬프트 본문에 포함시키세요. 
        아래는 형식이 아니라 개념 설명입니다 - 이 문장을 그대로 베끼지 말고, 
        실제 주어진 함수명과 파일 경로, 기능 설명을 사용해 당신만의 자연스러운 
        문장으로 새로 작성하세요.
        (개념 예: "회사 정보를 응답 객체로 바꾸는 기능이 이미 다른 파일에 구현되어 
        있다면, 새로 만들지 말고 그 함수를 가져다 써라"는 취지로 작성)
    - 복잡도 문제와 재사용 문제는 서로 다른 지시입니다. 복잡도 높은 함수는 
      "재사용하지 말라"가 아니라 "구조를 개선하라"고 지시하세요.
    - 복잡도가 높은 함수를 개선하라고 지시할 때, "구조를 개선하라"처럼 모호하게 
      끝내지 마세요. 반드시 다음 중 최소 하나를 구체적으로 명시하세요: 
      "각 등급/케이스별로 별도의 private 메서드로 분리하라", 
      "switch-case 문으로 변경하라", "조건을 테이블(Map) 기반으로 재구성하라" 등.
    - 같은 함수에 대한 재사용 지시를 두 번 이상 반복하지 마세요. 
      이미 앞에서 언급한 함수는 다시 설명하지 마세요.
    - "재사용 가능한 기존 함수" 목록에 없는 함수는 재사용 대상으로 언급하지 마세요.
    - 원본 프롬프트의 핵심 목적은 절대 바꾸지 마세요.
    - "안전하게", "적절히" 같은 모호한 표현은 금지합니다.
    - 입력된 문제점 목록 각각에 대해 빠짐없이 지시 문장을 추가하세요.
    - 이전 시도에 대한 피드백이 주어지면, 그 지적사항을 반드시 반영해서 다시 작성하세요.
    - 반드시 순수 JSON: {"reconstructed_prompt": "...", "explanation": "..."}
    """

    user_prompt = f"""
    [원본 프롬프트]
    {original_prompt}

    [원본 코드]
    {code}

    [발견된 문제점]
    {issues_text}
    {feedback_instruction}

    위 문제가 재발하지 않도록 원본 프롬프트를 재구성해주세요. 반드시 JSON으로만 응답하세요.
    """

    try:
        completion = client.chat.completions.create(
            model="google/diffusiongemma-26b-a4b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.1,
            max_tokens=2048,
            stream=False
        )
        print("[LOG] draft_reconstruct_prompt DiffusionGemma 응답 받음") 
        raw_output = completion.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw_output, strict=False)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            parsed = json.loads(match.group(0), strict=False) if match else {"reconstructed_prompt": original_prompt, "explanation": ""}

        return parsed
    except Exception as e:
        print(f"프롬프트 초안 작성 중 에러: {e}")
        return {"reconstructed_prompt": original_prompt, "explanation": "초안 작성에 실패했습니다."}


def review_prompt(draft: dict, issues_text: str) -> dict:
    """초안을 검토만 함. 문제 있으면 그 내용을 반환, 없으면 통과 처리"""
    print("[LOG] review_prompt 시작")
    
    system_prompt = """
당신은 프롬프트 품질 검수자입니다. 직접 수정하지 말고, 오직 평가만 하세요.
아래 [재구성된 프롬프트 초안]이 [발견된 문제점 목록]을 빠짐없이, 모순 없이 반영했는지 검토하세요.

[검토 기준]
1. 논리적 모순: "~할 수 없다"와 재사용 지시가 동시에 있는 등 앞뒤가 안 맞는 문장이 있는가?
2. 누락: [발견된 문제점 목록]에 있는 항목 중, 초안에서 언급 자체가 아예 빠진 게 있는가?
3. 재사용 지시가 있다면, 그 함수가 무엇을 반환하는지에 대한 최소한의 설명이 있는가?
4. 복잡도 개선 대상 함수에 "재사용하지 말라"는 식의 잘못된 지시가 섞여있지 않은가?

[중요 - 관대하게 판단할 것]
- 서로 다른 함수는 서로 다른 기능 설명을 가지는 것이 정상입니다. 
  "두 함수의 설명이 서로 다르다"는 이유만으로 실패 처리하지 마세요.
- 완벽하지 않아도, 위 4가지 기준을 최소한으로 충족하면 통과(passed: true) 처리하세요. 
  사소한 표현 방식의 차이나 문장 구조의 미묘함은 문제 삼지 마세요.
- 오직 명백하고 중대한 문제(모순, 완전한 누락, 잘못된 지시)만 실패로 판단하세요.

문제가 없으면 {"passed": true, "feedback": ""}로 응답하세요.
명백하고 중대한 문제가 있으면 {"passed": false, "feedback": "구체적으로 어떤 부분이 왜 문제인지 설명"}으로 응답하세요.
반드시 순수 JSON으로만 응답하세요.
"""

    user_prompt = f"""
    [발견된 문제점 목록]
    {issues_text}

    [검토할 프롬프트 초안]
    {draft.get('reconstructed_prompt', '')}

    [초안의 재구성 이유]
    {draft.get('explanation', '')}

    위 기준으로 검토 결과를 JSON으로 응답하세요.
    """

    try:
        completion = client.chat.completions.create(
            model="google/diffusiongemma-26b-a4b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.1,
            max_tokens=1024,
            stream=False
        )
        print("[LOG] review_prompt DiffusionGemma 응답 받음")
        raw_output = completion.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw_output, strict=False)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            parsed = json.loads(match.group(0), strict=False) if match else {"passed": True, "feedback": ""}

        return parsed
    except Exception as e:
        print(f"검수 중 에러 발생: {e}")
        return {"passed": True, "feedback": ""} 

def has_duplicate_paragraphs(text: str) -> bool:
    """단순하게, 같은 문단(줄바꿈 두 번으로 구분)이 반복되는지 기계적으로 체크"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return len(paragraphs) != len(set(paragraphs))

def reconstruct_prompt(original_prompt: str, code: str, vulnerabilities: list, 
                       complexity_details: list, duplicate_snippets: list = None,
                       max_retries: int = 2) -> dict:
    
    print("[LOG] reconstruct_prompt 시작")
    issues_text = ""
    if vulnerabilities:
        vuln_list = "\n".join(f"- {v.get('message', '')}" for v in vulnerabilities[:5])
        issues_text += f"\n[발견된 보안 취약점]\n{vuln_list}\n"
    if complexity_details:
        comp_list = "\n".join(f"- {c.get('message', '')}" for c in complexity_details[:5])
        issues_text += f"\n[발견된 복잡도/비효율 이슈]\n{comp_list}\n"

    if duplicate_snippets:
        dup_list = "\n\n".join(
            f"- 함수명: {d.get('function_name')}\n  위치: {d.get('file_path')}\n"
            f"  기능 설명: {summarize_function(d)}"
            for d in duplicate_snippets[:3]
        )
        issues_text += f"\n[재사용 가능한 기존 함수]\n{dup_list}\n"

    if not issues_text:
        issues_text = "\n(특별히 발견된 문제는 없었습니다.)\n"
    # 여기까지 추가된 부분

    best_draft = None
    feedback = None

    for attempt in range(max_retries + 1):
        draft = draft_reconstruct_prompt(original_prompt, code, issues_text, feedback)
        print(f"[LOG] {attempt+1}번째 시도 시작")
        if has_duplicate_paragraphs(draft.get("reconstructed_prompt", "")):
            print(f"[프롬프트 재구성] {attempt + 1}번째 시도: 중복 문단 감지, 재시도")
            feedback = "이전 시도에서 같은 문단이 여러 번 반복되었습니다. 각 지시사항은 정확히 한 번씩만 작성하세요."
            continue

        review = review_prompt(draft, issues_text)
        
        if review.get("passed", True):
            best_draft = draft
            print(f"[프롬프트 재구성] {attempt + 1}번째 시도에서 검수 통과")
            break
        
        best_draft = draft
        feedback = review.get("feedback", "")
        print(f"[프롬프트 재구성] {attempt + 1}번째 시도 검수 실패, 피드백: {feedback}")

    return best_draft if best_draft else {"reconstructed_prompt": original_prompt, "explanation": "재구성 실패"}

async def reconstruct_prompt_endpoint(request: PromptReconstructRequest) -> PromptReconstructResponse:
    result = reconstruct_prompt(
        request.original_prompt,
        request.code_content,
        request.vulnerabilities,
        request.complexity_details,
        request.duplicate_snippets
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
        patched_code=security_result["patched_code"],
        duplicate_snippets=request.duplicate_snippets or []
    )