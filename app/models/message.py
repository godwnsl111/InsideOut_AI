from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from app.database import Base

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    content = Column(Text)
    author_type = Column(String(10))  # "USER" 또는 "AI"
    created_at = Column(DateTime(timezone=True), server_default=func.now())