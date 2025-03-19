from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from app.database import Base

class Session(Base):
    __tablename__ = "session"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    is_closed = Column(Boolean, default=False)
    ors_score = Column(Integer, default=0)
    srs_score = Column(Integer, default=0)
    agreement = Column(String(10), default="DENIED")
    summary = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

