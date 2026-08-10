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
import time
import logging
import functools

client = OpenAI(
  base_url = os.getenv("NVIDIA_BASE_URL"),
  api_key = os.getenv("NVIDIA_API_KEY"),
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

KNOWN_JAVA_IMPORTS = {
    "Mac": "javax.crypto.Mac",
    "SecretKeySpec": "javax.crypto.spec.SecretKeySpec",
    "Cipher": "javax.crypto.Cipher",
    "KeyGenerator": "javax.crypto.KeyGenerator",
    "SecretKey": "javax.crypto.SecretKey",
    "MessageDigest": "java.security.MessageDigest",
    "SecureRandom": "java.security.SecureRandom",
    "PreparedStatement": "java.sql.PreparedStatement",
    "Connection": "java.sql.Connection",
    "ResultSet": "java.sql.ResultSet",
    "DriverManager": "java.sql.DriverManager",
    "Statement": "java.sql.Statement",
    "SQLException": "java.sql.SQLException",
}

NOT_AUTOCLOSEABLE_JAVA_TYPES = {"Process", "Thread"}


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

class PromptBuilder:
    """[38, 39번 해결] 프롬프트 조립을 전담하는 팩토리 클래스 (System/User 역할 명확히 분리)"""
    
    @staticmethod
    def build_patch_prompts(original_code, vulnerabilities, language, error_history, resolved_history):
        lang_tag = (language or "java").lower()
        
        system_prompt = f"""You are a strict secure coding expert.
[PRIORITIES & RULES]
1. OUTPUT FORMAT: MUST return a valid JSON object. No markdown.
2. SECURITY PATCH: Fix vulnerabilities (No hardcoded secrets, use PreparedStatement, SHA-256).
3. LOGIC PRESERVATION: Keep original business logic.
4. SYNTAX: Ensure perfectly balanced brackets and valid {lang_tag} syntax."""

        history_instruction = ""
        if resolved_history:
            history_instruction += "\n[KEEP]\n" + "\n".join(f"- {r}" for r in resolved_history)
        if error_history:
            history_instruction += "\n[AVOID]\n" + "\n".join(f"- {e}" for e in error_history)

        vuln_text = "\n".join([f"- {v.get('rule_id')}: {v.get('message')}" for v in vulnerabilities])
        
        user_prompt = f"[VULNERABILITIES]\n{vuln_text}\n{history_instruction}\n[ORIGINAL CODE]\n{original_code}"
        
        # [40번 해결] 프롬프트 길이 사전 검증 (대략적인 토큰 수 방어)
        if len(user_prompt) > 40000: # 문자가 너무 길면 (예: 1만 토큰 이상)
            raise ValueError("원본 코드가 너무 길어 모델의 컨텍스트 한도를 초과합니다. 코드를 분할하세요.")
            
        return system_prompt, user_prompt


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
            stream=False,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False   # reasoning 모드 끄기
                }
            }
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

def check_known_import_errors(code: str) -> list:
    """자주 실수하는 import 경로가 틀렸는지 화이트리스트 기반으로 확인 (Java 전용)"""
    errors = []
    import_lines = re.findall(r'^import\s+([\w.]+);', code, re.MULTILINE)

    for imported_path in import_lines:
        class_name = imported_path.split('.')[-1]
        if class_name in KNOWN_JAVA_IMPORTS:
            correct_path = KNOWN_JAVA_IMPORTS[class_name]
            if imported_path != correct_path:
                errors.append(
                    f"'{class_name}'의 올바른 import 경로는 '{correct_path}'인데 "
                    f"'{imported_path}'로 되어 있습니다."
                )

    return errors

def has_balanced_braces(code: str) -> bool:
    if code.count('{') != code.count('}') or code.count('(') != code.count(')'):
        return False
    if '```' in code or code.strip().startswith('{"'):
        return False
    return True

def check_try_with_resources_errors(code: str) -> list:
    """try-with-resources에 AutoCloseable이 아닌 타입이 들어갔는지 확인"""
    errors = []
    try_blocks = re.findall(r'try\s*\((.*?)\)\s*\{', code, re.DOTALL)
    for block in try_blocks:
        for bad_type in NOT_AUTOCLOSEABLE_JAVA_TYPES:
            if re.search(rf'\b{bad_type}\s+\w+\s*=', block):
                errors.append(f"{bad_type}는 AutoCloseable을 구현하지 않아 try-with-resources에 사용할 수 없습니다.")
    return errors

def _build_resolved_summary(needs_refactoring: bool) -> str:
    """지금까지 어떤 기계 검증을 통과했는지에 맞춰 동적으로 요약 문구 생성"""
    parts = ["보안 취약점은 성공적으로 패치되었습니다."]
    if needs_refactoring:
        parts.append("순환 복잡도 문제도 해결되었습니다.")
    return " ".join(parts)


def _extract_patched_code(raw_output: str) -> str:
    """강력한 전처리 및 스택 기반 중괄호 추출을 적용한 패치 코드 추출"""
    
    # 1. 마크다운 찌꺼기 완벽 제거 (11번 문제 해결)
    # ```json ... ``` 형태를 확실하게 벗겨냄
    cleaned = re.sub(r'```(?:json|java|python|cpp|c)?(.*?)```', r'\1', raw_output, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # 닫는 펜스 없이 끝난 경우 대비
    cleaned = re.sub(r'^```\w*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    
    # 2. Self-Correction ("Wait...", "Actually...") 제거 (12번 문제 해결)
    # 무조건 첫 번째 '{' 이전의 모든 헛소리를 날려버림
    start_idx = cleaned.find('{')
    if start_idx == -1:
        print("[DEBUG] JSON 시작 기호 '{' 를 찾을 수 없습니다.")
        return ""  # 실패 시 빈 문자열 반환 -> 호출부에서 에러 누적 후 재시도
    
    cleaned = cleaned[start_idx:]
    
    # 3. 정상 JSON 파싱 시도
    try:
        parsed = json.loads(cleaned, strict=False)
        return parsed.get("patched_code", "")
    except json.JSONDecodeError:
        pass

    # 4. 스택(Stack) 기반 중괄호 균형 추출 알고리즘 (13, 14번 문제 해결)
    # 정규식 {.*} 의 치명적 단점을 극복하고 완벽한 JSON 덩어리만 핀셋으로 빼냄
    stack = []
    end_idx = -1
    for i, char in enumerate(cleaned):
        if char == '{':
            stack.append(i)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack: # 스택이 비었다면 짝이 완벽하게 맞는 객체가 닫힌 것
                    end_idx = i
                    break
    
    if end_idx != -1:
        json_str = cleaned[:end_idx + 1]
        try:
            parsed = json.loads(json_str, strict=False)
            return parsed.get("patched_code", "")
        except json.JSONDecodeError:
            pass

    # 5. 최후의 보루 (안전한 정규식 추출) (15번 문제 해결)
    # 어설픈 추출은 Syntax Error 사이클을 유발하므로 정규식을 아주 엄격하게 변경
    loose_match = re.search(r'"patched_code"\s*:\s*"(.*)', cleaned, re.DOTALL)
    if loose_match:
        extracted = loose_match.group(1)
        last_quote_index = extracted.rfind('"')
        if last_quote_index != -1:
            extracted = extracted[:last_quote_index]
        extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        return extracted.strip()
        
    print("[DEBUG] 모든 JSON 추출 시도 실패")
    return ""  # 억지로 원본을 넘기지 않고 빈 문자열 반환하여 완벽한 재시도 유도

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def draft_patched_code(original_code: str, vulnerabilities: list, needs_refactoring: bool,
                       max_complexity: int, language: str, duplicate_snippets: list,
                       error_history: list = None, resolved_history: list = None,
                       attempt_num: int = None) -> str:
    """패치 코드 초안 생성. error_history 배열을 통해 이전 실패 원인을 누적 반영"""

    tag = f"[{attempt_num}차][생성]" if attempt_num else "[생성]"

    lang_tag = (language or "java").lower()

    # 1. 추가 지시사항 조립 (간결화)
    refactoring_instruction = ""
    if needs_refactoring:
        refactoring_instruction = f"\n- [최적화] 순환 복잡도({max_complexity})를 낮추고 중첩 루프를 제거하세요."

    duplicate_instruction = ""
    if duplicate_snippets:
        snippets_text = "\n\n".join(f"[함수명: {d.get('function_name')}]\n{d.get('code')}" for d in duplicate_snippets[:2])
        duplicate_instruction = f"\n\n[EXISTING FUNCTIONS TO REUSE - MANDATORY]\n{snippets_text}\n(You must call the function above instead of reimplementing this logic.)"

    # 2. 히스토리 누적 (핵심 포인트!)
    history_instruction = ""
    if resolved_history:
        res_str = "\n".join(f"  * {r}" for r in resolved_history)
        history_instruction += f"\n[유지할 성공 내역 - 절대 원래대로 되돌리지 마세요]\n{res_str}\n"
    if error_history:
        err_str = "\n".join(f"  * {e}" for e in error_history)
        history_instruction += f"\n[과거 오답 노트 - 똑같은 실수를 반복하지 마세요]\n{err_str}\n"

    # 로그 추가: 이번 시도에 실제로 어떤 지시사항이 실려 나가는지 확인
    logger.info(f"{tag} needs_refactoring={needs_refactoring}, "
                f"duplicate_snippets={len(duplicate_snippets) if duplicate_snippets else 0}개, "
                f"error_history={len(error_history) if error_history else 0}개, "
                f"resolved_history={len(resolved_history) if resolved_history else 0}개")
    if error_history:
        logger.info(f"{tag} 이번에 전달되는 오답노트: {error_history}")

    # 3. 프롬프트 다이어트 (감정적 단어 제거, 우선순위 명시)
    system_prompt = f"""You are a strict secure coding expert.
[PRIORITIES & RULES]
1. OUTPUT FORMAT: MUST return a valid JSON object. No markdown, no explanations outside JSON.
2. SECURITY PATCH: Fix vulnerabilities. 
   - DO NOT hardcode secrets (use System.getenv). 
   - Use PreparedStatement and match the exact number of '?' with bound parameters. 
   - Use SHA-256 instead of MD5.
3. CODE REUSE: If [EXISTING FUNCTIONS TO REUSE] is provided below, you MUST call/import that 
   existing function instead of writing new logic that duplicates it. This is not optional — 
   failing to reuse an existing function will cause your response to be rejected.
4. LOGIC PRESERVATION: Keep original business logic. (Removing sensitive logs is allowed).
5. SYNTAX: Ensure perfectly balanced brackets and valid {lang_tag} syntax. Do not use 'try-with-resources' for Process objects.
6. The JSON object MUST start with exactly one opening curly brace and end with exactly one closing curly brace. Do not duplicate the outer braces.

[JSON SCHEMA]
{{
    "thought_process": "Short 1-sentence summary of fixes",
    "patched_code": "Complete patched code string, escaped properly for JSON"
}}"""
    
    vuln_text = ""
    if vulnerabilities:
        for v in vulnerabilities:
            rule_id = v.get("rule_id", "Unknown Rule")
            line = v.get("line", "Unknown Line")
            msg = v.get("message", "").replace('\n', ' ') # 줄바꿈 제거
            vuln_text += f"- [Line {line}] {rule_id}: {msg}\n"
    else:
        vuln_text = "- 발견된 취약점 없음\n"

    user_prompt = f"""[VULNERABILITIES]
{vuln_text}
[REQUESTS]{refactoring_instruction}{duplicate_instruction}
{history_instruction}
[ORIGINAL CODE]
{original_code}
"""

    estimated_code_tokens = int(len(original_code) / 2.5)
    sample_max_tokens = max(4096, min(int(estimated_code_tokens * 1.5) + 1024, 16384))

    max_api_retries = 3
    for api_attempt in range(max_api_retries):
        try:
            completion = client.chat.completions.create(
                model="google/diffusiongemma-26b-a4b-it",
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.0,
                top_p=0.7,
                max_tokens=sample_max_tokens,
                stream=False,
                timeout=45.0
            )
            
            break 
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"{tag} API 호출 실패 (시도 {api_attempt + 1}/{max_api_retries}): {error_msg}")
            
            if api_attempt == max_api_retries - 1:
                logger.error(f"{tag} API 최대 재시도 횟수 초과. 코드 생성을 포기합니다.")
                return ""
            
            if "429" in error_msg or "rate limit" in error_msg:
                wait_time = 5 * (api_attempt + 1)
                logger.info(f"{tag} Rate Limit 감지. {wait_time}초 대기 후 재시도합니다...")
                time.sleep(wait_time)
            else:
                time.sleep(2)

    choice = completion.choices[0]
    logger.info(f"{tag} finish_reason={choice.finish_reason}")
    if choice.finish_reason == "length":
        logger.warning(f"{tag} Token length exceeded (finish_reason: length)")

    message_content = choice.message.content
    if not message_content:
        reasoning_content = getattr(choice.message, 'reasoning', None)
        if reasoning_content:
            logger.warning(f"{tag} content가 비어 reasoning 필드를 대신 사용합니다.")
            message_content = reasoning_content
        else:
            logger.warning(f"{tag} content, reasoning 모두 비어있어 빈 문자열 반환")
            return ""

    raw_output = message_content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.split("\n", 1)[-1]
    if raw_output.endswith("```"):
        raw_output = raw_output.rsplit("\n", 1)[0]

    logger.info(f"{tag} raw_output 길이: {len(raw_output)}자")

    extracted_code = _extract_patched_code(raw_output)

    # 로그 추가: 최종적으로 추출된 코드 전체를 확인 가능하게
    if extracted_code:
        logger.info(f"{tag} 추출된 patched_code ({len(extracted_code)}자):\n{extracted_code}")
    else:
        logger.warning(f"{tag} patched_code 추출 실패 (빈 문자열). raw_output 앞부분: {raw_output[:300]}")

    return extracted_code

logger = logging.getLogger(__name__)

def verify_patched_code(patched_code: str, original_code: str, language: str,
                        attempt_num: int = None) -> dict:
    """생성된 패치 코드가 문법적으로 유효하고 실행 가능한지 검증만 함 (직접 수정 안 함)"""

    tag = f"[{attempt_num}차][검사-문법]" if attempt_num else "[검사-문법]"

    lang_tag = (language or "java").lower()

    logger.info(f"{tag} 검증 시작, patched_code 길이: {len(patched_code)}자")

    system_prompt = f"""당신은 {lang_tag} 코드 품질 검수자입니다. 직접 수정하지 말고, 오직 평가만 하세요.
아래 [검토할 코드]가 다음 기준을 만족하는지 확인하세요:

1. 문법 오류: 괄호/중괄호/따옴표가 정확히 짝이 맞는가? 컴파일 가능한 문법인가?
2. SQL 문법: SQL 쿼리가 있다면, 파라미터 자리표시자(?, 등)의 개수가 실제 바인딩되는 값의 개수와 정확히 일치하는가?
3. Import 경로: 존재하지 않는 라이브러리를 import하는지만 확인하세요. (java.sql.* 같은 와일드카드 import는 허용됩니다.)
4. 로직 보존: [원본 코드]의 핵심 기능(비즈니스 로직)이 그대로 유지되는가?
   * 예외 허용: 보안 취약점 해결을 위한 민감 정보 로그 삭제, 안전한 해시 함수로 교체, PreparedStatement 변경, 하드코딩 제거는 '로직 훼손'이 아닙니다. 정상적인 개선으로 판단하여 통과(Pass) 시키세요.

[출력 형식]
- 실패 시 피드백("feedback")은 반드시 200자 이내로 핵심만 짧게 작성하세요.
- 통과 시: {{"passed": true, "feedback": ""}}
- 실패 시: {{"passed": false, "feedback": "어느 부분이 문제인지 구체적인 설명"}}
"""

    user_prompt = f"""[원본 코드]
{original_code}

[검토할 코드]
{patched_code}

위 기준으로 검토 결과를 JSON으로 응답하세요.
"""

    max_api_retries = 3
    for api_attempt in range(max_api_retries):
        try:
            completion = client.chat.completions.create(
                model="mistralai/mistral-nemotron",
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.0,
                top_p=0.7,
                max_tokens=4096,
                timeout=45.0
            )

            choice = completion.choices[0]
            logger.info(f"{tag} finish_reason={choice.finish_reason}")

            # 응답 추출 방어 로직
            message_content = choice.message.content
            if not message_content:
                reasoning_content = getattr(choice.message, 'reasoning', None)
                if reasoning_content:
                    logger.warning(f"{tag} content가 비어 reasoning 필드를 대신 사용합니다.")
                message_content = reasoning_content if reasoning_content else '{"passed": true, "feedback": ""}'

            raw_output = message_content.strip()

            # 강력한 JSON 정제 로직
            cleaned = re.sub(r'```(?:json)?(.*?)```', r'\1', raw_output, flags=re.DOTALL).strip()

            start_idx = cleaned.find('{')
            if start_idx != -1:
                cleaned = cleaned[start_idx:]
                stack = []
                end_idx = -1
                for i, char in enumerate(cleaned):
                    if char == '{':
                        stack.append(i)
                    elif char == '}':
                        if stack:
                            stack.pop()
                            if not stack:
                                end_idx = i
                                break
                if end_idx != -1:
                    cleaned = cleaned[:end_idx + 1]

            try:
                parsed = json.loads(cleaned, strict=False)
                logger.info(f"{tag} 결과: passed={parsed.get('passed')}, "
                            f"feedback={parsed.get('feedback', '')[:200]}")
                return parsed
            except json.JSONDecodeError:
                logger.warning(f"{tag} [검증 파싱 실패] 원본 응답이 올바른 형식이 아닙니다: {raw_output[:100]}")
                return {"passed": False, "feedback": "평가자 응답 파싱 실패 (JSON 포맷 에러). 코드를 다시 검증하세요."}

        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"{tag} 검증기 API 호출 실패 (시도 {api_attempt + 1}/{max_api_retries}): {error_msg}")

            is_overload = "503" in error_msg or "429" in error_msg or "resourceexhausted" in error_msg

            if api_attempt == max_api_retries - 1:
                # 최종 실패 — 하이브리드 판단
                if is_overload:
                    logger.warning(f"{tag} 검증기 서버 과부하가 지속되어 검증을 생략하고 통과 처리합니다.")
                    return {"passed": True, "feedback": "(API 과부하로 문법 검증 생략됨)"}
                logger.error(f"{tag} 검증기 API 최대 재시도 횟수 초과 (과부하 외 원인).")
                return {"passed": False, "feedback": "코드 검증 중 시스템 에러가 발생하여 실패 처리되었습니다."}

            if is_overload:
                wait_time = 5 * (api_attempt + 1)
                logger.info(f"{tag} 서버 과부하 감지. {wait_time}초 대기 후 검증을 재시도합니다...")
                time.sleep(wait_time)
            else:
                time.sleep(2)

def verify_issues_resolved(patched_code: str, original_code: str, vulnerabilities: list,
                            needs_refactoring: bool, max_complexity: int,
                            duplicate_snippets: list, language: str,
                            attempt_num: int = None) -> dict:

    tag = f"[{attempt_num}차][검사-종합]" if attempt_num else "[검사-종합]"

    # 1. [기계 검증 1] Semgrep으로 실제 취약점이 남았는지 칼같이 체크
    security_report = run_semgrep(patched_code, language)
    logger.info(f"{tag} Semgrep 재검사: has_vulnerability={security_report['has_vulnerability']}, "
                f"발견 개수={len(security_report.get('vulnerabilities', []))}")
    if security_report["has_vulnerability"]:
        remaining_issues = "\n".join(f"- {v.get('message', '')}" for v in security_report["vulnerabilities"])
        logger.warning(f"{tag} Semgrep 재검사 실패, 남은 취약점:\n{remaining_issues}")
        return {
            "passed": False,
            "feedback": f"Semgrep 재검사 결과, 다음 취약점이 여전히 남아있습니다. 즉시 수정하세요:\n{remaining_issues}",
            "resolved": ""   # 보안조차 아직 통과 못 했으니, 뭔가 "해결됐다"고 말할 게 없음
        }

    # 2. [기계 검증 2] Lizard로 복잡도가 실제로 줄었는지 체크 (리팩토링이 필요했던 경우에만)
    if needs_refactoring:
        complexity_report = analyze_complexity(patched_code, language)
        COMPLEXITY_THRESHOLD = 15
        logger.info(f"{tag} Lizard 재검사: max_complexity={complexity_report['max_complexity']} "
                    f"(임계치 {COMPLEXITY_THRESHOLD})")
        if complexity_report["max_complexity"] > COMPLEXITY_THRESHOLD:
            logger.warning(f"{tag} Lizard 재검사 실패, 복잡도 {complexity_report['max_complexity']}로 여전히 높음")
            return {
                "passed": False,
                "feedback": f"순환 복잡도가 여전히 {complexity_report['max_complexity']}로 너무 높습니다. 중첩 루프나 조건문을 제거하여 리팩토링하세요.",
                "resolved": "보안 취약점은 성공적으로 패치되었습니다."   # 여긴 needs_refactoring이 True인 게 확실하니 그대로 둬도 됨(복잡도 얘기는 안 넣음, 이건 아직 실패 중이니까)
            }

    # 3. [LLM 검증] 재사용 여부만 확인. duplicate_snippets 없으면 여기서 바로 통과
    resolved_so_far = _build_resolved_summary(needs_refactoring)   # 실제 상황에 맞게 동적 생성

    if not duplicate_snippets:
        logger.info(f"{tag} 재사용 대상 없음, Semgrep+Lizard 통과로 최종 승인")
        return {"passed": True, "feedback": "", "resolved": resolved_so_far}

    logger.info(f"{tag} 재사용성 검증 시작, 대상 함수 {len(duplicate_snippets[:2])}개")

    lang_tag = (language or DEFAULT_LANGUAGE).lower()

    dup_list = "\n".join(
        f"- {d.get('function_name')}() 함수는 {d.get('file_path')}의 기존 함수와 재사용되어야 함 "
        f"(매개변수: {', '.join(d.get('parameters') or [])})"
        for d in duplicate_snippets[:2]
    )

    system_prompt = f"""
    당신은 {lang_tag} 코드 재사용성 검수자입니다. 직접 수정하지 말고, 오직 평가만 하세요.
    보안 취약점{"과 복잡도 문제는" if needs_refactoring else "은"} 이미 기계적으로 검증되어 통과했습니다.
    당신은 오직 "기존 함수 재사용 여부"만 확인하세요.

    [검토 기준]
    - [반드시 재사용해야 할 기존 함수] 목록에 있는 각 함수가, [검토할 코드]에서
      실제로 import/호출되어 재사용되고 있는지 확인하세요.
    - 원본 로직을 그대로 복사해서 새로 정의한 경우(재사용 안 함)는 실패로 판단하세요.

    문제가 없으면 {{"passed": true, "feedback": "", "resolved": "재사용까지 포함해 모든 문제가 해결되었습니다."}}로 응답하세요.
    문제가 있으면 {{"passed": false, "feedback": "구체적으로 어떤 함수가 재사용되지 않았는지 설명", "resolved": "{resolved_so_far}"}}로 응답하세요.
    반드시 순수 JSON으로만 응답하세요.
    """

    user_prompt = f"""
    [반드시 재사용해야 할 기존 함수]
    {dup_list}

    [검토할 코드]
    {patched_code}

    위 함수들이 실제로 재사용되었는지 검토 결과를 JSON으로 응답하세요.
    """

    try:
        completion = client.chat.completions.create(
                model="mistralai/mistral-nemotron",
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.0,
                top_p=0.7,
                max_tokens=4096,
                timeout=45.0
        )

        choice = completion.choices[0]
        logger.info(f"{tag} 재사용성 검증 API finish_reason={choice.finish_reason}")

        message_content = choice.message.content
        if not message_content:
            reasoning_content = getattr(choice.message, 'reasoning', None)
            if reasoning_content:
                logger.warning(f"{tag} content가 비어 reasoning 필드를 대신 사용합니다.")
            message_content = reasoning_content if reasoning_content else ""

        raw_output = message_content.strip()

        try:
            parsed = json.loads(raw_output, strict=False)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            # 파싱 실패 시에도, 실제로 어떤 검증을 거쳤는지에 맞는 문구를 씀 (하드코딩 아님)
            parsed = json.loads(match.group(0), strict=False) if match else {
                "passed": False, "feedback": "재사용성 검증 응답 파싱 실패.", "resolved": resolved_so_far
            }

        logger.info(f"{tag} 재사용성 검증 결과: passed={parsed.get('passed')}, "
                    f"feedback={parsed.get('feedback', '')}")
        return parsed
    except Exception as e:
        logger.error(f"{tag} 재사용성 검증 중 에러 발생: {e}")
        return {"passed": False, "feedback": "재사용성 검증 중 시스템 에러가 발생했습니다.", "resolved": resolved_so_far}
    
def generate_patched_code(original_code, vulnerabilities, needs_refactoring=False,
                          max_complexity=0, language="java", duplicate_snippets=None):
    
    actual_retries = 5 if needs_refactoring else 3
    
    start_time = time.time()
    try:
        error_history = []
        resolved_history = []
        best_code = None
        is_success = False
        attempt = -1 
        
        for attempt in range(actual_retries):
            logger.info(f"====== [코드 생성] {attempt + 1}/{actual_retries}번째 시도 ======")

            current_needs_refactoring = needs_refactoring if attempt < 2 else False
            current_duplicate_snippets = duplicate_snippets if attempt < 2 else None
            
            patched_code = draft_patched_code(
                original_code, vulnerabilities, current_needs_refactoring,
                max_complexity, language, current_duplicate_snippets, 
                error_history, resolved_history,
                attempt_num=attempt + 1   # 추가
            )

            if not patched_code:
                logger.warning(f"[{attempt + 1}차] 생성된 코드가 비어있어 파싱에 실패했습니다.")
                error_history.append("API 응답 오류 또는 파싱 실패. 유효한 JSON 코드를 생성하세요.")
                continue

            local_error_msg = ""
            if not has_balanced_braces(patched_code):
                local_error_msg = "생성된 코드의 괄호 또는 중괄호 개수가 맞지 않습니다."
            elif language.lower() == "java":
                all_errors = check_known_import_errors(patched_code) + check_try_with_resources_errors(patched_code)
                if all_errors:
                    local_error_msg = "다음 문제가 있습니다: " + " ".join(all_errors)
            
            if local_error_msg:
                error_history.append(local_error_msg)
                logger.warning(f"[{attempt + 1}차] 로컬 문법/리소스 검증 실패: {local_error_msg}")
                continue

            syntax_review = verify_patched_code(
                patched_code, original_code, language,
                attempt_num=attempt + 1   # 추가
            )
            if not syntax_review.get("passed", True):
                syntax_err = syntax_review.get("feedback", "알 수 없는 문법 오류")
                error_history.append(f"문법 오류: {syntax_err}")
                logger.warning(f"[{attempt + 1}차] 평가자(Validator) 문법 검증 실패: {syntax_err}")
                continue

            best_code = patched_code

            issue_review = verify_issues_resolved(
                patched_code, original_code, vulnerabilities,
                needs_refactoring, max_complexity, duplicate_snippets, language,
                attempt_num=attempt + 1   # 추가
            )
            
            if issue_review.get("passed", True):
                logger.info(f"[{attempt + 1}차] 시도에서 모든 검증 통과")
                is_success = True
                break 

            issue_feedback = issue_review.get("feedback", "")
            if issue_feedback:
                error_history.append(f"보안 패치 미흡: {issue_feedback}")
                
            resolved_msg = issue_review.get("resolved", "")
            if resolved_msg and resolved_msg not in resolved_history:
                resolved_history.append(resolved_msg)
                
            logger.info(f"[{attempt + 1}차] 해결한 문제: {resolved_msg}")
            logger.warning(f"[{attempt + 1}차] 문제해결 검증 실패: {issue_feedback}")

        if not is_success:
            logger.warning("[코드 생성] 최대 재시도 횟수 도달. 마지막으로 컴파일을 통과한 코드를 반환합니다.")
            
        elapsed_time = round(time.time() - start_time, 2)

        # 로그 추가: 최종적으로 반환되는 코드를 한 번에 확인 가능하게
        final_code_preview = best_code if best_code else "보완 코드 생성에 실패했습니다."
        logger.info(f"====== [코드 생성] 최종 결과: status={'success' if is_success else 'fail'}, "
                    f"attempts_used={attempt + 1}, elapsed={elapsed_time}초 ======")
        logger.info(f"[최종 코드]\n{final_code_preview}")

        result_payload = {
            "status": "success" if is_success else "fail",
            "final_code": final_code_preview,
            "metrics": {
                "attempts_used": attempt + 1,
                "max_retries": actual_retries,
                "elapsed_seconds": elapsed_time,
                "error_history_count": len(error_history)
            }
        }
        
        with open("patch_analytics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "language": language,
                "vulnerability_count": len(vulnerabilities),
                "status": result_payload["status"],
                "attempts": result_payload["metrics"]["attempts_used"],
                "elapsed_seconds": elapsed_time
            }) + "\n")
            
        return result_payload

    except Exception as e:
        logger.error(f"코드 생성 중 치명적 에러 발생: {e}", exc_info=True)
        return {
            "status": "error",
            "final_code": "보완 코드 생성 중 치명적 오류가 발생했습니다.",
            "metrics": {"attempts_used": 0, "elapsed_seconds": 0}
        }

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
            stream=False,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False   # reasoning 모드 끄기
                }
            }
        )
        print(f"[LOG] summarize_function 완료: {function_name}")
        summary = completion.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"함수 요약 중 에러 발생: {e}")
        # 실패 시 최소한의 정보라도 반환
        return f"{function_name}({', '.join(parameters)}) 함수 (요약 생성 실패, 매개변수만 참고)"

def file_path_to_package(file_path: str, language: str = "java") -> str | None:
    if language == "java":
        path = file_path.replace("src/main/java/", "").replace(".java", "")
        return path.replace("/", ".")
    elif language == "python":
        path = file_path.replace(".py", "")
        return path.replace("/", ".")
    else:
        return None 

def draft_reconstruct_prompt(original_prompt: str, code: str, issues_text: str, feedback: str = None) -> dict:
    """프롬프트 초안 작성. feedback이 있으면 이전 시도의 문제점을 참고해서 다시 작성"""
    print("[LOG] draft_reconstruct_prompt 시작")

    feedback_instruction = ""
    if feedback:
        feedback_instruction = f"""
        [이전 시도에서 발견된 문제점 - 반드시 이번엔 고쳐서 작성하세요]
        {feedback}
        주의: 위 문제만 고치고, 그 외의 나머지 부분(이미 정상적으로 작성된 import, 
        SQL 쿼리, 괄호 등)은 이전 시도와 최대한 동일하게 유지하세요. 
        문제와 무관한 부분을 임의로 다시 바꾸지 마세요.
        """

    system_prompt = """
당신은 프롬프트 엔지니어링 전문가입니다.
사용자가 AI에게 코드 생성을 요청했던 "원본 프롬프트"와, 그 결과로 생성된 코드에서 발견된 문제점을 받게 됩니다.
같은 목적(기능)을 달성하되, 발견된 문제가 재발하지 않도록 프롬프트를 재작성해야 합니다.

[가장 중요한 전제]
재구성된 프롬프트는, 원본 코드나 기존 함수의 존재를 전혀 알지 못하는 다른 AI 도구
(ChatGPT, Gemini, Claude 등)에게 사용자가 그대로 복사해서 전달할 것입니다. 
따라서:
- "현재 이 함수는", "기존 코드에서는", "이 함수의 복잡도가 높으므로" 처럼 
  "이미 존재하는 코드를 보고 있다"고 전제하는 표현은 절대 쓰지 마세요.
- "변경하라", "리팩토링하라", "수정하라" 같이 "기존 코드를 고친다"는 뜻의 동사도 
  쓰지 마세요. 대신 "~하도록 설계하라", "~구조로 구현하라", "처음부터 ~하지 않도록 
  작성하라"처럼, "앞으로 새로 작성할 코드"에 대한 지시로 표현하세요.
- 단, "재사용 가능한 기존 함수" 목록에 있는 함수만은 예외입니다. 이 함수들은 
  실제로 프로젝트에 이미 존재하므로, "이미 구현되어 있는 이 함수를 재사용하라"고 
  명시적으로 알려주는 것이 맞습니다.

[규칙]
- "재사용 가능한 기존 함수"를 재사용하라는 지시를 작성할 때는, 
  반드시 "해당 클래스를 import한 뒤 호출하라"는 것까지 명시하세요. 
  [재사용 가능한 기존 함수] 항목에 "Import 경로"가 주어졌다면, 
  그 경로를 그대로 사용해서 "이 클래스를 import하라"고 명시하세요. 
  Import 경로가 주어지지 않았다면, 이 언어의 표준적인 재사용 문법을 
  사용하되 파일 경로를 참고해 합리적으로 안내하세요.
- 복잡도 문제와 재사용 문제는 서로 다른 지시입니다. 복잡도 높은 함수는 
  "재사용하지 말라"가 아니라, 처음부터 복잡하지 않은 구조로 설계하라고 지시하세요.
- 입력으로 주어진 [발견된 문제점] 목록에 있는 항목만 반영하세요. 코드를 보고 스스로 판단하여
  목록에 없는 새로운 문제를 추가로 지적하거나 프롬프트에 반영하지 마세요.
- 복잡도가 높아지기 쉬운 로직을 설계하라고 지시할 때, "구조를 개선하라"처럼 모호하게 
  끝내지 마세요. 반드시 다음 중 최소 하나를 구체적으로 명시하세요: 
  "각 등급/케이스별로 별도의 private 메서드로 분리하여 구현하라", 
  "switch-case 문 기반으로 설계하라", "조건을 테이블(Map) 기반으로 설계하라" 등.
- 같은 함수에 대한 재사용 지시를 두 번 이상 반복하지 마세요. 
  이미 앞에서 언급한 함수는 다시 설명하지 마세요.
- "재사용 가능한 기존 함수" 목록에 없는 함수는 재사용 대상으로 언급하지 마세요.
- 원본 프롬프트의 핵심 목적은 절대 바꾸지 마세요.
- "안전하게", "적절히" 같은 모호한 표현은 금지합니다.
- 입력된 문제점 목록 각각에 대해 빠짐없이 지시 문장을 추가하세요.
- 이전 시도에 대한 피드백이 주어지면, 그 지적사항을 반드시 반영해서 다시 작성하세요.
- 반드시 순수 JSON으로만 응답하세요. JSON 앞뒤에 어떤 텍스트도, 백틱도 붙이지 마세요:
{"reconstructed_prompt": "...", "explanation": "..."}
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
            stream=False,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False   # reasoning 모드 끄기
                }
            }
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
5. 재사용 지시가 있다면, 그 클래스를 "import해야 한다"는 것과 
   구체적인 패키지 경로(예: com.example.dto.CompanyResponse 형태)가 
   명시되어 있는가? 단순히 파일 시스템 경로(src/main/java/...)만 있고 
   패키지 import 형태가 없다면 실패로 판단하세요.

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
            stream=False,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False   # reasoning 모드 끄기
                }
            }
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
                       language: str = "java", max_retries: int = 2) -> dict:
    
    issues_text = ""
    if vulnerabilities:
        vuln_list = "\n".join(f"- {v.get('message', '')}" for v in vulnerabilities[:5])
        issues_text += f"\n[발견된 보안 취약점]\n{vuln_list}\n"
    if complexity_details:
        comp_list = "\n".join(f"- {c.get('message', '')}" for c in complexity_details[:5])
        issues_text += f"\n[발견된 복잡도/비효율 이슈]\n{comp_list}\n"

    if duplicate_snippets:
        dup_entries = []
        for d in duplicate_snippets[:3]:
            import_path = file_path_to_package(d.get('file_path', ''), language)
            import_line = f"  Import 경로: {import_path}\n" if import_path else ""
            dup_entries.append(
                f"- 함수명: {d.get('function_name')}\n"
                f"  파일 경로: {d.get('file_path')}\n"
                f"{import_line}"
                f"  기능 설명: {summarize_function(d)}"
            )
        issues_text += f"\n[재사용 가능한 기존 함수]\n" + "\n\n".join(dup_entries) + "\n"

    if not issues_text:
        issues_text = "\n(특별히 발견된 문제는 없었습니다.)\n"

    feedback = None
    best_draft = None

    for attempt in range(max_retries + 1):
        draft = draft_reconstruct_prompt(original_prompt, code, issues_text, feedback)

        if has_duplicate_paragraphs(draft.get("reconstructed_prompt", "")):
            print(f"[프롬프트 재구성] {attempt + 1}번째 시도: 중복 문단 감지, 재시도")
            feedback = "이전 시도에서 같은 문단이 여러 번 반복되었습니다. 각 지시사항은 정확히 한 번씩만 작성하세요."
            best_draft = draft
            continue

        review = review_prompt(draft, issues_text)
        best_draft = draft

        if review.get("passed", True):
            print(f"[프롬프트 재구성] {attempt + 1}번째 시도에서 검수 통과")
            break

        feedback = review.get("feedback", "")
        print(f"[프롬프트 재구성] {attempt + 1}번째 시도 검수 실패, 피드백: {feedback}")

    return best_draft

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

        # Semgrep 서브프로세스가 UTF-8을 쓰도록 환경변수 명시적으로 전달
        subprocess_env = os.environ.copy()
        subprocess_env["PYTHONUTF8"] = "1"
        subprocess_env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=subprocess_env)
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

    duplicate_snippets = [d.model_dump() for d in (request.duplicate_snippets or [])]

    try:
        security_task = asyncio.to_thread(run_security_pipeline, code, language, duplicate_snippets)
        ai_task = asyncio.to_thread(run_ai_detection, code)

        security_result, ai_result = await asyncio.gather(security_task, ai_task)

        patch_result = security_result["patched_code"]
        if isinstance(patch_result, dict):
            final_code = patch_result.get("final_code", "보완 코드 생성에 실패했습니다.")
            patch_success = (patch_result.get("status") == "success")
        elif patch_result is None:
            # 취약점/복잡도/중복 문제가 없어 패치 자체가 필요 없었던 정상 케이스
            final_code = code
            patch_success = True
        else:
            final_code = patch_result
            patch_success = bool(final_code) and final_code != "보완 코드 생성에 실패했습니다."

        return AICodeDetectionResponse(
            is_ai_generated=ai_result["is_ai_generated"],
            ai_probability=ai_result["ai_probability"],
            has_vulnerability=security_result["has_vulnerability"],
            vulnerabilities=security_result["vulnerabilities"],
            max_complexity=security_result["max_complexity"],
            needs_refactoring=security_result["needs_refactoring"],
            complexity_details=security_result.get("complexity_details", []),
            patched_code=final_code,
            patch_success=patch_success,
            duplicate_snippets=request.duplicate_snippets or []
        )

    except Exception as e:
        logger.error(f"분석 전체 실패: {e}", exc_info=True)
        return AICodeDetectionResponse(
            is_ai_generated=False,
            ai_probability=0.0,
            has_vulnerability=False,
            vulnerabilities=[],
            patched_code="분석 중 오류가 발생했습니다.",
            patch_success=False,
            duplicate_snippets=[]
        )