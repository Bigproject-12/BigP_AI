from fastapi import FastAPI
from app.api import analysis_router

app = FastAPI(title="BIGP AI 서버")

# 라우터 등록 (주소가 http://localhost:8000/api/ai/detect 가 됩니다)
app.include_router(analysis_router.router, prefix="/api/ai", tags=["AI Code Analysis"])

@app.get("/")
def root():
    return {"message": "AI 서버가 정상적으로 실행 중입니다!"}