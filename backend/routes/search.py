from fastapi import APIRouter
from rag.embedder import vector_store

router = APIRouter()

@router.get("/search")
async def search_debug(q: str, top_k: int = 5):
    results = vector_store.search(q, top_k)
    return {"query": q, "results": results}
