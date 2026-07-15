import os
import boto3
import torch
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


async def detect_ai_code(request: AICodeDetectionRequest) -> AICodeDetectionResponse:
    # 스프링 부트(또는 포스트맨)에서 전달받은 원본 코드 내용
    code = request.code_content
    
    # max_length=512: 모델이 한 번에 읽을 수 있는 최대 길이로 커팅
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    
    # 💡 4. AI 모델 추론 시작
    with torch.no_grad(): # backpropagation(학습 역전파)을 막아서 메모리와 속도를 극대화합니다.
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
        ai_probability=round(ai_prob, 2) # 소수점 둘째 자리까지만
    )