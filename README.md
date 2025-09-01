# redfin\_rag — FastAPI 기반 RAG + 뉴스 출간 API

AI 관련 기사/문서를 대상으로 **RAG(Retrieval-Augmented Generation)** 파이프라인과 **자동 뉴스 출간 기능**을 제공하는 백엔드입니다.
1차 기능은 `redfin_target-insight`, 확장 기능은 `redfin_news`입니다.

---

## 핵심 요약

* **서버**: FastAPI (Uvicorn)
* **임베딩**: BGE-base (`BAAI/bge-base-en-v1.5`)
* **VectorStore**: **ChromaDB** (cosine, 기본), 실험용 FAISS 인덱스 지원
* **Retriever**: `as_retriever(k|fetch_k|lambda_mult|filter)` with MMR
* **LLM**: .env 설정에 따라 플러그블 (예: OpenAI gpt-4.1-mini, Gemini 등)
* **CORS**: `ALLOWED_ORIGINS=["*"]` 기본
* **MongoDB**:

  * `redfin.rag_logs` — 모든 `/redfin_target-insight` 요청/응답 자동 저장
  * `redfin.news_posts` — 뉴스 출간 결과 저장
* **뉴스 출간**:

  * 템플릿: `prompts/news.py` (Smart Brevity 한국어 고정)
  * JSON 파싱/보증: `schemas/news_llm.py` (Pydantic) + 번역 레이어
* **주요 엔드포인트**:

  * `POST /redfin_target-insight` (RAG 질의)
  * `POST /redfin_news/publish_from_env` (뉴스 출간 → Mongo 저장)
  * `GET /redfin_news/posts`, `GET /redfin_news/posts/{post_id}`
  * `GET /healthz` (헬스체크)

---

## 디렉터리 구조(발췌)

```
src/
├─ api_rag.py                # FastAPI 엔트리포인트
├─ core/
│  ├─ settings.py            # 환경변수 로딩
│  └─ lifespan.py            # startup/shutdown 훅 (인덱스+Mongo 초기화)
├─ routers/
│  ├─ redfin.py              # /redfin_target-insight, /logs/save
│  └─ news.py                # /redfin_news/* 라우트
├─ services/
│  ├─ rag_service.py         # RAG 쿼리 실행 + 뉴스용 날짜필터(ts)
│  ├─ news_service.py        # 뉴스 출간, Pydantic 파싱 + 한국어 보증
│  └─ strategy.py            # 검색/LLM 전략 선택
├─ nureongi/
│  ├─ loaders.py             # 데이터 로더 (published_at_ts 메타 추가)
│  ├─ indexing.py            # 청크/인덱싱 파이프라인
│  ├─ vectorstore.py         # Chroma/FAISS VectorStore
│  ├─ chain.py               # 리트리버 + 생성 체인
│  └─ raptor.py              # RAPTOR 요약 트리
├─ prompts/
│  └─ news.py                # Smart Brevity 한국어 템플릿
├─ observability/
│  ├─ mongo_logger.py        # MongoDB 로깅 유틸
│  └─ langsmith.py           # LangSmith 연동
├─ schemas/
│  ├─ query.py               # 요청 모델
│  ├─ response.py            # 응답 모델
│  ├─ news.py                # NewsPublishRequest/NewsPost
│  └─ news_llm.py            # LLM 출력 파싱 모델
├─ tesst/
│  └─ test_news_view.html    # 단건 뉴스 뷰어
├─ test_rag_client.html      # RAG 테스트 클라이언트 (자동 Mongo 로깅)
├─ .chroma/                  # Chroma 퍼시스턴스 디렉터리
└─ faiss_index/              # FAISS 인덱스(옵션)
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
ALLOWED_ORIGINS=["*"]

# VectorStore
VECTORSTORE_PROVIDER=chroma
CHROMA_DIR=./src/.chroma
FAISS_DIR=./src/faiss_index

# RAPTOR
RAPTOR_ENABLED=true
RAPTOR_TARGET_K=3
RAPTOR_INDEX_MODE=summary_only

# MongoDB
MONGO_URI=mongodb://192.168.0.123:27017
MONGO_DB=redfin
MONGO_COL=rag_logs
NEWS_MONGO_COL=news_posts

# 뉴스 출간
NEWS_API_URL=http://.../feed.json
NEWS_FEED_FIELD_MAP={"title":"title","content":"article_text","url":"link","id":"guid"}
NEWS_DEFAULT_PUBLISH=1
NEWS_TOP_K=6
NEWS_RECENCY_DAYS=14
```

---

## 실행

### 로컬

```bash
pip install -r requirements.txt
cd src
python api_rag.py
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

### 1) `POST /redfin_target-insight`

```json
{
  "question": "최근 LLM 경량화 동향을 요약해줘.",
  "top_k": 5,
  "fetch_k": 20,
  "lambda_mult": 0.5,
  "persona": "ai_industry_professional"
}
```

응답은 자동으로 `redfin.rag_logs`에 기록.

---

### 2) `POST /redfin_news/publish_from_env`

뉴스 피드에서 자동 기사 생성 후 `redfin.news_posts`에 저장.

```bash
curl -X POST "http://localhost:8001/redfin_news/publish_from_env"
```

---

### 3) `GET /redfin_news/posts`

```bash
curl "http://localhost:8001/redfin_news/posts?limit=1"
```

---

## 테스트 클라이언트

### `test_rag_client.html`

* 요청 보내기 → API 응답 확인
* 자동으로 MongoDB에 로깅됨 (수동 저장 버튼 제거됨)

### `tesst/test_news_view.html`

* 최신 기사 1건 또는 `?post_id=` 지정 조회 가능

```bash
python -m http.server 5500
# http://localhost:5500/tesst/test_news_view.html
```

---

## 로깅/관측

* **자동 저장**: `/redfin_target-insight`, `/redfin_news/publish*` 응답 모두 MongoDB 기록
* **LangSmith 연동**: `.env`에 `LANGSMITH_PROJECT=API_RAG` 설정 시 트레이싱

---

## 개발 체크리스트

* [x] `/redfin_target-insight` → 자동 Mongo 로깅
* [x] `/redfin_news/publish_from_env` → 뉴스 출간 후 DB 저장
* [x] `test_news_view.html`에서 TL;DR + 본문 렌더링
* [ ] TL;DR 3줄 준수율 강화 (리프로그램 추가 예정)
* [ ] 인덱스 기동 최적화 (백그라운드 로딩)
* [ ] 로그 공통 스키마 정리
* [ ] SSE 스트리밍 응답

---

## 업데이트 내역

* 2025-08-27: `api_rag.py` 최초 구현, MongoDB 자동 저장 적용
* 2025-08-28: `redfin_news` 파이프라인 추가, Smart Brevity 한국어 템플릿 추가
* 2025-08-28: 기사 출간 언어 문제 분석
 

