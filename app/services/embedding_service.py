# app/services/embedding_service.py
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tree_sitter_language_pack import get_parser
from tree_sitter import Query, QueryCursor

EMBED_MODEL_PATH = "./app/models/codebert-base" # 모델 경로
FAISS_INDEX_DIR = "./app/faiss_indexes" # 임베딩 파일 경로
DIMENSION = 768 # CODE BERT 모델이 처리할 수 있는 차원 고정값

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH) # 코드를 모델이 이해하기 위한 토큰으로 변환하기 위한 도구
embed_model = AutoModel.from_pretrained(EMBED_MODEL_PATH) # 가중치
embed_model.eval() # 학습이 아닌 추론을 하기 위한 작업 학습엔 .train 으로 사용

os.makedirs(FAISS_INDEX_DIR, exist_ok=True) # 파일 경로가 없으면 만들기 위함
print("임베딩 모델 로딩 완료")


# 파일 확장자 -> 언어명 매핑 (언어 감지용)
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".html": "html",
}

# 언어명 -> tree-sitter-languages가 인식하는 언어 문자열 매핑
TREE_SITTER_LANGUAGE_MAP = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
    "c": "c",
    "csharp": "c_sharp",
    "cpp": "cpp",
}

# 함수 단위 청킹을 지원하는 언어별 tree-sitter 쿼리 패턴
# (여기 없는 언어는 함수 개념이 없거나 아직 미지원 -> 슬라이딩 윈도우로 자동 폴백)
TREE_SITTER_QUERY_PATTERNS = {
    "python": "(function_definition name: (identifier) @func_name) @func",
    "java": "(method_declaration name: (identifier) @func_name) @func",
    "javascript": "(function_declaration name: (identifier) @func_name) @func",
    "c": "(function_definition declarator: (function_declarator declarator: (identifier) @func_name)) @func",
    "csharp": "(method_declaration name: (identifier) @func_name) @func",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier) @func_name)) @func",
}


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
    """기존 슬라이딩 윈도우 방식 청킹 (함수 단위 파싱이 안 되는 언어, 또는 파싱 실패 시 폴백용)"""
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


def detect_language_from_path(file_path: str) -> str:
    """파일 확장자로 언어를 추론. 매칭 안 되면 'unknown' 반환"""
    for ext, lang in EXTENSION_TO_LANGUAGE.items():
        if file_path.endswith(ext):
            return lang
    return "unknown"


def chunk_code_by_function(code_content: str, file_path: str, language: str) -> list[dict]:
    """Tree-sitter로 함수/메서드 단위 청킹 + 함수명/매개변수 추출"""
    parser = get_parser(TREE_SITTER_LANGUAGE_MAP[language])
    code_bytes = code_content.encode("utf-8")
    tree = parser.parse(code_bytes)

    query_str = TREE_SITTER_QUERY_PATTERNS[language]
    query = Query(parser.language, query_str)
    cursor = QueryCursor(query)                    # 추가
    captures = cursor.captures(tree.root_node)      # query. → cursor.

    # captures는 {capture_name: [node, ...]} 형태의 딕셔너리로 반환됨
    func_nodes = captures.get("func", [])

    chunks = []
    for node in func_nodes:
        name_node = node.child_by_field_name("name")
        function_name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8") if name_node else "unknown"

        params_node = node.child_by_field_name("parameters")
        parameters = []
        if params_node:
            for child in params_node.children:
                if child.type in ("identifier", "typed_parameter", "formal_parameter", "parameter"):
                    param_text = code_bytes[child.start_byte:child.end_byte].decode("utf-8")
                    parameters.append(param_text)

        func_code = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        chunks.append({
            "file_path": file_path,
            "function_name": function_name,
            "parameters": parameters,
            "start_line": start_line,
            "end_line": end_line,
            "code": func_code
        })

    if not chunks:
        raise ValueError(f"{file_path}에서 함수를 찾지 못했습니다.")

    return chunks


def chunk_code_smart(code_content: str, file_path: str, language: str) -> list[dict]:
    """언어에 따라 함수 단위 청킹 또는 슬라이딩 윈도우 청킹을 선택"""
    if language in TREE_SITTER_QUERY_PATTERNS:
        try:
            return chunk_code_by_function(code_content, file_path, language)
        except Exception as e:
            print(f"{language} 함수 단위 파싱 실패, 슬라이딩 윈도우로 폴백: {e}")
            return chunk_code(code_content, file_path)
    else:
        return chunk_code(code_content, file_path)


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
        language = detect_language_from_path(f["file_path"])
        chunks = chunk_code_smart(f["content"], f["file_path"], language)
        for chunk in chunks:
            vector = get_embedding(chunk["code"])
            index.add_with_ids(vector.reshape(1, -1), np.array([vector_id_counter], dtype="int64"))
            metadata_result.append({
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "faiss_vector_id": vector_id_counter,
                "function_name": chunk.get("function_name"),
                "parameters": chunk.get("parameters", []),
                "code": chunk["code"]
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


def remove_vectors(repo_id: int, vector_ids: list[int]) -> int:
    """특정 레포의 FAISS 인덱스에서 지정된 벡터 ID들을 제거하고, 제거된 개수를 반환"""
    path = _index_path(repo_id)
    if not os.path.exists(path):
        # 인덱스 자체가 없으면 지울 것도 없음
        return 0

    index = faiss.read_index(path)

    ids_to_remove = np.array(vector_ids, dtype="int64")
    removed_count = index.remove_ids(ids_to_remove)  # FAISS가 실제 제거된 개수를 반환함

    save_index(repo_id, index)
    return removed_count