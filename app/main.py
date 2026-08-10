from dotenv import load_dotenv
load_dotenv()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from app.api import analysis_router, embedding_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BIGP AI 서버")

# 라우터 등록
app.include_router(analysis_router.router, prefix="/api/ai", tags=["AI Code Analysis"])
app.include_router(embedding_router.router, prefix="/api/embedding", tags=["Code Embedding"])

origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGIN", "http://localhost:5173").split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI 서버가 정상적으로 실행 중입니다!"}
