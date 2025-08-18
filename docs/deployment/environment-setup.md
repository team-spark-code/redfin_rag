# 환경 설정 가이드

이 문서는 RAG 시스템의 개발 및 운영 환경 설정 방법을 안내합니다.

## 개요

RAG 시스템을 실행하기 위해 필요한 환경 설정과 의존성 설치 방법을 단계별로 설명합니다.

## 시스템 요구사항

### 최소 요구사항
- **운영체제**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.10 이상
- **메모리**: 8GB RAM (권장 16GB)
- **저장공간**: 10GB 이상의 여유 공간
- **네트워크**: 인터넷 연결 (모델 다운로드용)

### 권장 사양
- **CPU**: 4코어 이상
- **GPU**: NVIDIA GPU (CUDA 지원, 선택사항)
- **메모리**: 16GB RAM
- **저장공간**: SSD 50GB 이상

## 설치 단계

### 1. Python 환경 설정

#### Python 설치 확인
```bash
python --version
# Python 3.10.0 이상이어야 합니다
```

#### 가상환경 생성
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate
```

### 2. 의존성 설치

#### 기본 패키지 설치
```bash
# requirements.txt 설치
pip install -r requirements.txt
```

#### 추가 패키지 (선택사항)
```bash
# 개발 도구
pip install black pytest pytest-cov

# GPU 지원 (NVIDIA GPU가 있는 경우)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 환경변수 설정

#### 환경변수 파일 생성
```bash
# 예시 파일 복사
cp env.example .env
```

#### 필수 환경변수 설정
```bash
# .env 파일 편집
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

### 4. 데이터 준비

#### 데이터셋 폴더 생성
```bash
# dataset 폴더가 없다면 생성
mkdir -p dataset
```

#### PDF 파일 추가
```bash
# 분석할 PDF 파일을 dataset/ 폴더에 복사
cp your_documents.pdf dataset/
```

## 환경별 설정

### 개발 환경

#### 개발용 환경변수 설정
```bash
# env.development 파일 사용
cp env.development .env
```

#### 개발 서버 실행
```bash
# 개발 모드로 서버 실행
python main.py
```

### 프로덕션 환경

#### 프로덕션용 환경변수 설정
```bash
# env.production 파일 사용
cp env.production .env
```

#### 프로덕션 서버 실행
```bash
# 프로덕션 모드로 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Docker 환경 설정

### Docker 설치
```bash
# Docker Desktop 설치 (Windows/macOS)
# https://www.docker.com/products/docker-desktop

# Docker 설치 (Ubuntu)
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

### Docker 이미지 빌드
```bash
# Docker 이미지 빌드
docker build -t rag-system .

# 컨테이너 실행
docker run -p 8000:8000 rag-system
```

## 문제 해결

### 자주 발생하는 문제

#### 1. Python 버전 오류
```bash
# Python 3.10 이상 설치 확인
python --version

# 가상환경 재생성
rm -rf venv
python -m venv venv
```

#### 2. 패키지 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 캐시 클리어
pip cache purge

# 개별 패키지 설치
pip install langchain faiss-cpu sentence-transformers
```

#### 3. 메모리 부족 오류
```bash
# 배치 크기 줄이기
export EMBEDDING_BATCH_SIZE=8

# GPU 사용 (가능한 경우)
export EMBEDDING_DEVICE=cuda
```

#### 4. API 키 오류
```bash
# 환경변수 확인
echo $OPENAI_API_KEY

# .env 파일 확인
cat .env
```

## 성능 최적화

### 메모리 최적화
```bash
# 청크 크기 조정
export CHUNK_SIZE=300
export CHUNK_OVERLAP=30

# 배치 크기 조정
export EMBEDDING_BATCH_SIZE=16
```

### GPU 최적화 (선택사항)
```bash
# CUDA 설치 확인
nvidia-smi

# GPU 사용 설정
export EMBEDDING_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0
```

## 보안 설정

### API 키 보안
```bash
# .env 파일 권한 설정
chmod 600 .env

# .gitignore에 .env 추가 확인
echo ".env" >> .gitignore
```

### 방화벽 설정
```bash
# 포트 8000 방화벽 허용 (필요시)
sudo ufw allow 8000
```

## 모니터링 설정

### 로그 설정
```bash
# 로그 레벨 설정
export LOG_LEVEL=INFO

# 로그 파일 경로 설정
export LOG_FILE=./logs/rag_system.log
```

### 헬스체크 설정
```bash
# 헬스체크 활성화
export ENABLE_HEALTH_CHECK=true
export HEALTH_CHECK_INTERVAL=30
```

## 체크리스트

### 설치 완료 확인
- [ ] Python 3.10 이상 설치됨
- [ ] 가상환경 생성 및 활성화됨
- [ ] requirements.txt 설치 완료
- [ ] 환경변수 파일 생성됨
- [ ] API 키 설정 완료
- [ ] 데이터셋 폴더 생성됨
- [ ] PDF 파일 추가됨

### 실행 확인
- [ ] `python main.py` 실행 성공
- [ ] 벡터 인덱스 생성 완료
- [ ] API 서버 정상 실행
- [ ] 테스트 실행 성공

## 참고 자료

- [Python 공식 문서](https://docs.python.org/)
- [Docker 공식 문서](https://docs.docker.com/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)

---

**마지막 업데이트**: 2024-07-12  
**작성자**: 우성민  
**문서 버전**: v1.0 