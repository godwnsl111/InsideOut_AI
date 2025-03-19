from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chatbot import router
from app.routes.group_feedback import router as group_feedback
from app.routes.recom_diag_summ import router as diag_router

app = FastAPI()

# CORS 설정 추가
origins = [
    "http://localhost:8080",  # Spring 서버
    "http://localhost:3000",   # 프론트엔드 서버
    "https://insideout-front.netlify.app/",
    "https://insideout-back-production.up.railway.app/",
    "https://insideout-back.azurewebsites.net",
    "https://emotionhq.com/"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router)
app.include_router(group_feedback)
app.include_router(diag_router)

@app.get("/")
async def root():
    return {"message": "챗봇 API가 실행 중입니다! ver0.0.2.12"}