from fastapi import APIRouter
from app.schemas.analysis_schema import (
    AICodeDetectionRequest, AICodeDetectionResponse,
    PromptReconstructRequest, PromptReconstructResponse
)
from app.services.analysis_service import detect_ai_code, reconstruct_prompt_endpoint

router = APIRouter()

@router.post("/detect", response_model=AICodeDetectionResponse)
async def detect_code_origin(request: AICodeDetectionRequest):
    return await detect_ai_code(request)

@router.post("/reconstruct-prompt", response_model=PromptReconstructResponse)
async def reconstruct(request: PromptReconstructRequest):
    return await reconstruct_prompt_endpoint(request)