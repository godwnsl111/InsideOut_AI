import os
from supabase import create_client, Client
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Supabase 클라이언트 초기화
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def get_db():
    return supabase

# SQLAlchemy Base 클래스 - 모델 정의용
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base() 