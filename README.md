# redfin_rag - FastAPI 기반 RAG API

Redfin 내부 지식을 대상으로 **Retrieval-Augmented Generation(RAG)** 답변을 제공하는 FastAPI 백엔드입니다. 뉴스 출간/분류 등 부가 기능은 제거되어 `/redfin_target-insight` 엔드포인트만 제공합니다.

---

## 주요 특징

- **서버**: FastAPI + Uvicorn
- **임베딩**: `BAAI/bge-base-en-v1.5`
- **VectorStore**: 기본 Chroma(cosine), 필요 시 FAISS 사용 가능
- **Retriever**: LangChain `as_retriever` 기반 MMR 검색
- **LLM**: OpenAI 또는 Google Generative AI(환경 변수로 선택)
- **관측**: MongoDB 로그 적재, LangSmith 트레이싱 지원

---

## 디렉터리 구조

```
src/
├── api_rag.py           # FastAPI 엔트리포인트
├── core/
│   ├── app.py           # FastAPI 앱 생성 및 CORS 설정
│   ├── lifespan.py      # Mongo 연결 + RAG 인덱스 초기화
│   └── settings.py      # Pydantic 기반 환경설정 로더
├── routers/
│   └── redfin.py        # POST /redfin_target-insight 라우터
├── schemas/
│   ├── query.py         # QueryRequest 스키마
│   └── response.py      # QueryResponseV1 스키마
├── services/
│   ├── rag_service.py   # 인덱스 준비 및 질의 실행
│   └── strategy.py      # 질문 의도 기반 검색 파라미터 결정
├── nureongi/            # 로더·인덱싱·체인 유틸리티 모음
└── observability/
    ├── mongo_logger.py  # MongoDB 로깅 헬퍼
    └── langsmith.py     # LangSmith 트레이서 구성
```

테스트 도구: `test/test_persona_rag.html`로 브라우저에서 API 응답을 빠르게 확인할 수 있습니다.

---

## 환경 변수(.env 예시)

```dotenv
# LLM 백엔드 키
OPENAI_API_KEY=...
# 또는
GOOGLE_API_KEY=...

# 임베딩 모델
EMB_MODEL=BAAI/bge-base-en-v1.5

# 서버 설정
HOST=0.0.0.0
PORT=8001
ALLOWED_ORIGINS=["*"]

# VectorStore
CHROMA_DIR=./src/.chroma
FAISS_DIR=./src/faiss_index

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGO_DB=redfin
MONGO_COL=rag_logs
MONGO_TIMEOUT_MS=3000

# RAG 소스 로딩
NEWS_INGEST_SOURCE=mongo       # 또는 http
NEWS_SOURCE_COLLECTION=extract # ingest_source=mongo일 때 사용
NEWS_API_URL=http://...        # ingest_source=http일 때 사용
```

---

## 실행 방법

```bash
pip install -r requirements.txt
cd src
python api_rag.py
```

- 헬스 체크: `GET http://localhost:8001/healthz`
- 문서 UI: `GET http://localhost:8001/docs`

Docker를 사용할 경우 `docker build -t redfin_rag .` 후 `docker run --env-file .env -p 8001:8001 redfin_rag`로 실행합니다.

---

## API

### `POST /redfin_target-insight`

```json
{
  "question": "최근 LLM 컴팩트 모델 동향을 요약해줘.",
  "top_k": 8,
  "fetch_k": 60,
  "lambda_mult": 0.25,
  "persona": "ai_industry_professional",
  "strategy": "auto"
}
```

- 응답은 LangChain 체인을 거쳐 생성되며 MongoDB(`rag_logs`)에 기록됩니다.
- 페르소나, 전략, 출처 메타데이터가 함께 반환됩니다.

헬스 체크: `GET /healthz`는 `{ "ok": true }`를 반환합니다.

---

## 관측 및 모니터링

- **MongoDB**: `rag_logs` 컬렉션에 요청/응답 메타데이터 저장
- **LangSmith**: `LANGCHAIN_PROJECT_REDFIN_TARGET` 환경 변수로 프로젝트명 지정
- **PII 마스킹**: `ENABLE_PII_REDACTION=true` 설정 시 이메일/전화번호/숫자를 단순 마스킹

---

## 개발 메모

- 서비스 기동 시 `core.lifespan`에서 Mongo 연결 후 `services.rag_service.init_index_auto()`로 벡터 인덱스를 준비합니다.
- `services.strategy.choose_strategy_advanced`가 질문 의도에 따라 검색 범위를 자동 조정합니다.
- 뉴스 출간/분류 관련 파일은 제거되었으며, RAG 기능만 유지됩니다.

필요한 설정 변경은 `.env` 또는 `core/settings.py`를 수정한 뒤 서버를 재시작하세요.

---

## 업데이트 내역
2025-08-27: api_rag.py 최초 구현, MongoDB 자동 저장 적용
2025-08-28: redfin_news 파이프라인 추가, Smart Brevity 한국어 템플릿 추가
2025-09-01: core/settings.py Pydantic v2 서브모델 구조 적용
2025-09-01: 뉴스 컬렉션 news_semantic_v1로 통일, 시드 제어 옵션(NEWS__SEED_ON_STARTUP)
2025-09-02: LangSmith 프로젝트명 기능별 분리(redfin_target-insight, redfin_news-publish)
2025-09-02: 프롬프트 관리 구조 개선(prompts/personas/*.md, system_insight.md)
2025-09-24: redfin_news 관련 코드 제거 및 RAG 전용 API로 정리
2025-09-24: 환경 변수 네이밍/한글 주석 정비 및 README/.env UTF-8 재작성
