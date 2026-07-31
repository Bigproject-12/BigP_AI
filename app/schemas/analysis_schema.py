from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class DuplicateSnippet(BaseModel):
    file_path: str
    function_name: Optional[str] = None
    parameters: Optional[List[str]] = Field(default_factory=list)
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    code: Optional[str] = None
    similarity_score: Optional[float] = None


class AICodeDetectionRequest(BaseModel):
    code_content: str = Field(..., description="검증할 원본 코드 내용")
    language: str = Field(default="java", description="분석할 코드의 프로그래밍 언어 (java, python, javascript, typescript 등)")
    duplicate_snippets: Optional[List[DuplicateSnippet]] = Field(
        default_factory=list,
        description="Spring이 미리 검색/보강해서 넘겨준 유사 코드 목록"
    )


class AICodeDetectionResponse(BaseModel):
    is_ai_generated: bool = Field(..., description="AI가 작성했을 것으로 판단되는지 여부 (True/False)")
    ai_probability: float = Field(..., description="AI가 작성했을 확률 (0.0 ~ 100.0%)")
    has_vulnerability: bool
    vulnerabilities: List[Dict[str, Any]]
    max_complexity: Optional[int] = 0
    needs_refactoring: Optional[bool] = False
    complexity_details: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="복잡도 임계치를 초과한 함수들의 상세 내역 (함수명, 점수, 라인, 메시지)")
    patched_code: Optional[str] = None
    duplicate_snippets: Optional[List[DuplicateSnippet]] = Field(default_factory=list)

class PromptReconstructRequest(BaseModel):
    original_prompt: str = Field(..., description="사용자가 이 코드를 생성할 때 실제로 사용한 프롬프트")
    code_content: str = Field(..., description="그 프롬프트로 생성된 원본 코드")
    vulnerabilities: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="이 코드에서 발견된 보안 취약점 목록")
    complexity_details: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="이 코드에서 발견된 복잡도/비효율 이슈 목록")


class PromptReconstructResponse(BaseModel):
    reconstructed_prompt: str = Field(..., description="문제점이 반영되어 재구성된 프롬프트")
    explanation: str = Field(..., description="어떤 부분을 왜 개선했는지에 대한 설명")