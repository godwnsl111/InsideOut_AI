# Step 1: Python 3.10 Slim (Debian 기반) 사용
FROM python:3.12-slim

# Step 2: 시스템 패키지 설치 (torch & scikit-learn 빌드용)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 3: 작업 디렉토리 설정
WORKDIR /app

# Step 4: 최신 pip 설치
RUN pip install --upgrade pip

# Step 5: 의존성 설치 (pip 패키지 설치)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: 애플리케이션 코드 복사
COPY . .

# Step 7: 포트 노출
ENV PORT=8000
EXPOSE 8000

# Step 8: FastAPI 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
