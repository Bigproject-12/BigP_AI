from fastapi import APIRouter
from app.schemas.analysis_schema import AICodeDetectionRequest, AICodeDetectionResponse
from app.services.analysis_service import detect_ai_code

router = APIRouter()

@router.post("/detect", response_model=AICodeDetectionResponse)
async def detect_code_origin(request: AICodeDetectionRequest):
    return await detect_ai_code(request)