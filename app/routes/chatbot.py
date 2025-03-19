from fastapi import APIRouter
from pydantic import BaseModel
from app.service.chatbot_service import process_rag_ors_chat

router = APIRouter()

class ChatRequest(BaseModel):
    sessionId: int
    user_input: str | None = None

@router.post("/api/process")
async def chat(request: ChatRequest):
    """
    세션 ID만 받아도 처리 가능하도록 수정.
    """
    try:
        response = await process_rag_ors_chat(str(request.sessionId), request.user_input or "")
        return response
    except Exception as e:
        return {"error": str(e)}