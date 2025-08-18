# RAG 시스템 Docker 이미지
# 
# 이 Dockerfile은 RAG(Retrieval-Augmented Generation) 시스템을 위한
# 컨테이너 이미지를 빌드합니다.
# 
# 주요 구성:
# - Python 3.10+ 기반
# - FastAPI 웹 서버
# - FAISS 벡터 데이터베이스
# - 한국어 임베딩 모델
# 
# 빌드 명령어: docker build -t rag-system .
# 실행 명령어: docker run -p 8000:8000 rag-system

# 베이스 이미지 설정
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
