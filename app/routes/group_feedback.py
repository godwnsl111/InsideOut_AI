from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.group_feedback_service import feedback_pipeline

router = APIRouter()

class FeedbackRequest(BaseModel):
    sessionIds: list[int]

@router.post("/api/department/improvements")
async def group_feedback(request: FeedbackRequest):
    """
    세션 ID 리스트를 받아 개선 사항을 요약하여 반환.
    """
    try:
        summary = feedback_pipeline(request.sessionIds)
        return {"improvements": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")