from pydantic import BaseModel, Field

class AICodeDetectionRequest(BaseModel):
    code_content: str = Field(..., description="검증할 원본 코드 내용")

class AICodeDetectionResponse(BaseModel):
    is_ai_generated: bool = Field(..., description="AI가 작성했을 것으로 판단되는지 여부 (True/False)")
    ai_probability: float = Field(..., description="AI가 작성했을 확률 (0.0 ~ 100.0%)")