from fastapi import APIRouter
from app.schemas.embedding_schema import (
    IndexRepoRequest, IndexRepoResponse,
    SearchDuplicateRequest, SearchDuplicateResponse
)
from app.services.embedding_service import index_repo_files, search_similar_code

router = APIRouter()

@router.post("/index", response_model=IndexRepoResponse)
async def index_repo(request: IndexRepoRequest):
    files = [{"file_path": f.file_path, "content": f.content} for f in request.files]
    chunk_metadata = index_repo_files(request.repo_id, files)
    return IndexRepoResponse(indexed_chunks=len(chunk_metadata), chunks=chunk_metadata)

@router.post("/search", response_model=SearchDuplicateResponse)
async def search_duplicates(request: SearchDuplicateRequest):
    results = search_similar_code(request.repo_id, request.code_content)
    return SearchDuplicateResponse(duplicates=results)