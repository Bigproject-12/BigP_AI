from fastapi import APIRouter
from app.schemas.embedding_schema import (
    IndexRepoRequest, IndexRepoResponse,
    SearchDuplicateRequest, SearchDuplicateResponse,
    RemoveVectorsRequest, RemoveVectorsResponse,
)
from app.services.embedding_service import index_repo_files, search_similar_code, remove_vectors 

router = APIRouter()

@router.post("/index", response_model=IndexRepoResponse)
async def index_repo(request: IndexRepoRequest):
    files = [{"file_path": f.file_path, "content": f.content} for f in request.files]
    chunk_metadata = index_repo_files(request.repo_id, files)
    return IndexRepoResponse(indexed_chunks=len(chunk_metadata), chunks=chunk_metadata)

@router.post("/search", response_model=SearchDuplicateResponse)
async def search_duplicates(request: SearchDuplicateRequest):
    results = search_similar_code(request.repo_id, request.code_content, request.language)
    return SearchDuplicateResponse(duplicates=results) 

@router.post("/remove", response_model=RemoveVectorsResponse)
async def remove_vectors_endpoint(request: RemoveVectorsRequest):
    removed_count = remove_vectors(request.repo_id, request.vector_ids)
    return RemoveVectorsResponse(removed_count=removed_count)