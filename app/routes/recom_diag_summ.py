from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.recom_diag_summ_service import diagnosis

# FastAPI 라우터 생성
router = APIRouter()

class DiagnosisRequest(BaseModel):
    sessionId: int

@router.post("/api/session/summary")
async def diagnose(request: DiagnosisRequest):
    """
    상담내용을 바탕으로 내담자의 상태 및 개선사항을 정리합니다.
    """
    try:
        # diagnosis 함수 호출
        response = diagnosis(int(request.sessionId))

        # 응답 반환
        return response

    except Exception as e:
        # 오류 발생 시 HTTPException으로 처리
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")