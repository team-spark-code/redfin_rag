# redfin\_rag — FastAPI 기반 RAG API

AI 관련 기사/문서를 대상으로 **RAG(Retrieval-Augmented Generation)** 파이프라인을 제공하는 백엔드입니다.
1차 구현 기능은 `redfin_target-insight`이며, 동일 파이프라인 위에 **자동 기사 출간** 기능(`redfin_news`)을 확장할 예정입니다.

---

## 핵심 요약

* **서버**: FastAPI (Uvicorn)
* **임베딩**: BGE-base (`BAAI/bge-base-en-v1.5`)
* **VectorStore**: **ChromaDB** (기본), 실험용 FAISS 인덱스 지원
* **Retriever**: `as_retriever(k|fetch_k|lambda_mult|filter)`
* **LLM**: .env 설정에 따라 플러그블 (예: OpenAI gpt-4.1-mini, Gemini 등)
* **CORS**: 프론트 로컬 `http://localhost:5500` 접근 허용
* **MongoDB**: `redfin.rag_logs`에 요청/응답 JSON 저장 (자동 + 수동 저장 버튼)
* **주요 엔드포인트**:

  * `POST /redfin_target-insight` (포트 기본 8001)
  * `POST /logs/save` (테스트 HTML에서 “몽고DB에 저장” 버튼으로 수동 저장)

---

## 디렉터리 구조(발췌)

```
src/
├─ api_rag.py                # FastAPI 엔트리포인트 (uvicorn 실행)
├─ core/                     # 공통 설정/라이프사이클
│  ├─ settings.py            # 환경변수 로딩
│  └─ lifespan.py            # startup/shutdown 훅 (인덱스+Mongo 초기화)
├─ routers/                  # 라우터 계층
│  └─ redfin.py              # /redfin_target-insight, /logs/save 라우트
├─ services/
│  ├─ rag_service.py         # RAG 쿼리 실행, 입력 정규화/후처리
│  └─ strategy.py            # 검색/LLM 전략 선택
├─ nureongi/                 # RAG 내부 모듈
│  ├─ loaders.py             # 데이터 로더
│  ├─ indexing.py            # 청크/인덱싱 파이프라인
│  ├─ vectorstore.py         # Chroma/FAISS VectorStore 생성
│  ├─ chain.py               # 리트리버 + 생성 체인
│  └─ raptor.py              # RAPTOR 요약 트리
├─ observability/
│  ├─ mongo_logger.py        # MongoDB(pymongo) 저장
│  └─ langsmith.py           # LangSmith 연동
├─ schemas/
│  ├─ query.py               # 요청 모델
│  └─ response.py            # 응답 모델
├─ .chroma/                  # Chroma 퍼시스턴스 디렉터리
├─ faiss_index/              # FAISS 인덱스(옵션)
└─ test_rag_client.html      # 테스트 클라이언트 (fetch + 수동 Mongo 저장 버튼)
```

---

## 환경 변수(.env)

```dotenv
# LLM 백엔드
LLM_BACKEND=openai
OPENAI_MODEL=gpt-4.1-mini
GEMINI_MODEL=gemini-2.0-flash

# 임베딩
EMB_MODEL=BAAI/bge-base-en-v1.5

# 서버
HOST=0.0.0.0
PORT=8001

# VectorStore
VECTORSTORE_PROVIDER=chroma   # chroma | faiss
CHROMA_DIR=./src/.chroma
FAISS_DIR=./src/faiss_index

# RAPTOR
RAPTOR_ENABLED=true
RAPTOR_TARGET_K=3
RAPTOR_INDEX_MODE=summary_only

# MongoDB
MONGODB_URI=mongodb+srv://<USER>:<PASS>@<CLUSTER>.mongodb.net/?retryWrites=true&w=majority
MONGO_DB=redfin
MONGO_COL=rag_logs
```

---

## 실행

### 로컬

```bash
pip install -r requirements.txt
cd src
python api_rag.py   # uvicorn 자동 실행
```

* 헬스체크: `GET http://localhost:8001/healthz`
* 문서 UI: `GET http://localhost:8001/docs`

### Docker (선택)

```bash
docker build -t redfin_rag .
docker run --env-file .env -p 8001:8001 redfin_rag
```

---

## API

### 1) `POST /redfin_target-insight` — RAG 질의 응답

#### Request 예시

```json
{
  "question": "최근 LLM 경량화 동향을 요약해줘.",
  "top_k": 5,
  "fetch_k": 20,
  "lambda_mult": 0.5,
  "persona": "ai_industry_professional"
}
```

#### Response 예시

```json
{
  "version": "v1",
  "data": {
    "answer": { "text": "마크다운 형식의 답변", "bullets": null, "format": "markdown" },
    "persona": "ai_industry_professional",
    "strategy": "map_refine",
    "sources": []
  },
  "meta": {
    "user": { "user_id": "demo-user", "session_id": "uuid-1234" },
    "request": { "service": "redfin_target-insight", "question": "...", "top_k": 5 },
    "pipeline": { "index_mode": "summary_only", "use_raptor": true, "embedding_model": "BAAI/bge-base-en-v1.5" }
  }
}
```

* 응답은 자동으로 `redfin.rag_logs`에 insert
* 로그 구조: `{ ts, endpoint, status, envelope, extra }`

### 2) `POST /logs/save` — 수동 로그 저장

테스트 HTML의 **“몽고DB에 저장”** 버튼에서 호출:

```json
{
  "envelope": { "version": "v1", "data": { ... } },
  "status": 200,
  "endpoint": "/redfin_target-insight",
  "extra": { "source": "test_client_manual_save" }
}
```

---

## 테스트 클라이언트 (test\_rag\_client.html)

* `요청 보내기` → API 응답 확인
* `몽고DB에 저장` → 응답 JSON을 `/logs/save`로 전송, MongoDB에 기록

테스트 방법:

```bash
uvicorn src.api_rag:app --reload --port 8001
python -m http.server 5500
# http://localhost:5500/test_rag_client.html
```

Mongo 확인:

```bash
mongosh "mongodb+srv://<USER>:<PASS>@<CLUSTER>.mongodb.net/redfin"
> db.rag_logs.find().sort({$natural:-1}).limit(1).pretty()
```

---

## 로깅/관측

* **자동 저장**: 모든 `/redfin_target-insight` 응답이 MongoDB에 기록
* **수동 저장**: `/logs/save`를 통해 강제로 insert 가능
* **LangSmith 연동**: `.env`에 `LANGSMITH_PROJECT=API_RAG` 설정 시 트레이스 추적 가능

---

## 개발 체크리스트

* [x] `/redfin_target-insight` 정상 응답 + Mongo 로그 자동 저장
* [x] `/logs/save` 수동 저장 버튼 동작
* [x] core/routers 분리 구조 적용
* [ ] RAPTOR hybrid 모드 검증
* [ ] LangSmith RunID → Mongo 로그 연계
* [ ] SSE 스트리밍 응답
* [ ] `POST /redfin_news` (자동 기사 출간 기능) 구현

---

## 업데이트 내역

* 2025-08-27: `api_rag.py` 최초 구현, MongoDB 수동 저장 기능 추가, core/routers 분리
* 2025-08-28: README.md 수정, MongoDB 자동 저장 기능 추가
