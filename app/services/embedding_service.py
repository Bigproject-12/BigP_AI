# app/services/embedding_service.py
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

EMBED_MODEL_PATH = "./app/models/codebert-base" # 모델 경로
FAISS_INDEX_DIR = "./app/faiss_indexes" # 임베딩 파일 경로
DIMENSION = 768 # CODE BERT 모델이 처리할 수 있는 차원 고정값

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH) # 코드를 모델이 이해하기 위한 토큰으로 변환하기 위한 도구
embed_model = AutoModel.from_pretrained(EMBED_MODEL_PATH) # 가중치
embed_model.eval() # 학습이 아닌 추론을 하기 위한 작업 학습엔 .train 으로 사용

os.makedirs(FAISS_INDEX_DIR, exist_ok=True) # 파일 경로가 없으면 만들기 위함
print("임베딩 모델 로딩 완료")


def get_embedding(code: str) -> np.ndarray: # 코드를 벡터로 변환하는 함수 N차원 배열을 다루는 클래스로 반환하겠단 뜻
    inputs = embed_tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
    # 토큰이저의 기본 사용 방식으로 code를 AI가 이해할 수 있는 숫자 단위의 토큰으로 변환하는 함수
    with torch.no_grad():
        outputs = embed_model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts
        vector = mean_pooled.squeeze(0).numpy().astype("float32")
    faiss.normalize_L2(vector.reshape(1, -1))
    return vector


def chunk_code(code: str, file_path: str, window_lines: int = 50, overlap: int = 10) -> list[dict]:
    lines = code.split("\n")
    chunks = []
    step = window_lines - overlap
    for start in range(0, len(lines), step):
        end = min(start + window_lines, len(lines))
        snippet = "\n".join(lines[start:end])
        if snippet.strip():
            chunks.append({
                "file_path": file_path,
                "start_line": start + 1,
                "end_line": end,
                "code": snippet
            })
        if end == len(lines):
            break
    return chunks


def _index_path(repo_id: int) -> str:
    return os.path.join(FAISS_INDEX_DIR, f"repo_{repo_id}.index")


def load_or_create_index(repo_id: int) -> faiss.IndexIDMap:
    path = _index_path(repo_id)
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexIDMap(faiss.IndexFlatIP(DIMENSION))


def save_index(repo_id: int, index: faiss.IndexIDMap):
    faiss.write_index(index, _index_path(repo_id))


def index_repo_files(repo_id: int, files: list[dict]) -> list[dict]:
    index = load_or_create_index(repo_id)
    vector_id_counter = index.ntotal
    metadata_result = []

    for f in files:
        chunks = chunk_code(f["content"], f["file_path"])
        for chunk in chunks:
            vector = get_embedding(chunk["code"])
            index.add_with_ids(vector.reshape(1, -1), np.array([vector_id_counter], dtype="int64"))
            metadata_result.append({
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "faiss_vector_id": vector_id_counter
            })
            vector_id_counter += 1

    save_index(repo_id, index)
    return metadata_result


def search_similar_code(repo_id: int, code: str, top_k: int = 5, threshold: float = 0.85) -> list[dict]:
    path = _index_path(repo_id)
    if not os.path.exists(path):
        return []

    index = faiss.read_index(path)
    query_vector = get_embedding(code).reshape(1, -1)
    scores, ids = index.search(query_vector, top_k)

    results = []
    for score, vec_id in zip(scores[0], ids[0]):
        if vec_id == -1:
            continue
        if score >= threshold:
            results.append({"faiss_vector_id": int(vec_id), "similarity_score": float(score)})
    return results


