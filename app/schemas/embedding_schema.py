from pydantic import BaseModel, Field
from typing import List, Dict, Any


class IndexFileItem(BaseModel):
    file_path: str = Field(..., description="레포 내 파일 경로")
    content: str = Field(..., description="파일 전체 소스 코드")


class IndexRepoRequest(BaseModel):
    repo_id: int = Field(..., description="GithubRepo 테이블의 repo_id")
    files: List[IndexFileItem] = Field(..., description="색인할 파일 목록")


class IndexRepoResponse(BaseModel):
    indexed_chunks: int = Field(..., description="새로 색인된 청크 개수")
    chunks: List[Dict[str, Any]] = Field(..., description="청크 메타데이터 (file_path, start_line, end_line, faiss_vector_id)")


class SearchDuplicateRequest(BaseModel):
    repo_id: int = Field(..., description="검색 대상 레포의 repo_id")
    code_content: str = Field(..., description="유사도를 검사할 코드")


class SearchDuplicateResponse(BaseModel):
    duplicates: List[Dict[str, Any]] = Field(..., description="유사 코드 청크 목록 (faiss_vector_id, similarity_score)")

class RemoveVectorsRequest(BaseModel):
    repo_id: int = Field(..., description="벡터를 지울 대상 레포의 repo_id")
    vector_ids: List[int] = Field(..., description="FAISS 인덱스에서 제거할 벡터 ID 목록")


class RemoveVectorsResponse(BaseModel):
    removed_count: int = Field(..., description="실제로 제거된 벡터 개수")