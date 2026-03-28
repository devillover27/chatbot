from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from models.schemas import ChatRequest
from rag.retriever import stream_rag_response
import json

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def event_generator():
        try:
            async for token in stream_rag_response(request.query, request.conversation_history):
                yield {
                    "event": "message",
                    "data": json.dumps({"text": token})
                }
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"})
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())
