아래 내용을 그대로 `README.md`에 붙여 넣으면 됩니다.
(현재 레포 스냅샷과 진행·계획을 반영했습니다. 필요 시 포트/엔드포인트/옵션만 바꿔 쓰면 됩니다.)

---

# redfin\_rag — FastAPI 기반 RAG API

AI 관련 기사/문서를 대상으로 **RAG(Retrieval-Augmented Generation)** 파이프라인을 제공하는 백엔드입니다.
1차 구현 기능은 `redfin_target-insight`이며, 동일 파이프라인 위에 **자동 기사 출간** 기능(`redfin_news`)을 확장할 예정입니다.

---

## 핵심 요약

* **서버**: FastAPI (Uvicorn)
* **임베딩**: BGE-base (`BAAI/bge-base-en-v1.5`)
* **VectorStore**: Chroma (기본), 실험용 FAISS 인덱스 보유
* **Retriever**: `as_retriever(k|fetch_k|lambda_mult|filter)`
* **LLM**: 프로젝트 .env 설정에 따라 플러그블 (예: OpenAI/Gemini 등)
* **CORS**: 프론트 로컬 `http://localhost:5500` 접근 허용
* **주요 엔드포인트**: `POST /redfin_target-insight` (포트 기본 8001)

---

## 디렉터리 구조(발췌)

```
.
├─ .env                         # 런타임 환경변수
├─ .env.example
├─ Dockerfile
├─ requirements.txt
├─ README.md
├─ docs/                        # 문서/메모
├─ scripts/                     # 배포/유틸 스크립트
└─ src/
   ├─ api_rag.py                # ★ FastAPI 앱(메인) — RAG API 라우트 정의
   ├─ api_classify.py           # 분류 API(별도, 선택)
   ├─ categorize_news.py        # 카테고리화 유틸/잡
   ├─ emb_test.py               # 임베딩 실험 스크립트
   ├─ main.py                   # 샘플/실험 엔트리
   ├─ test.py                   # 로컬 테스트
   ├─ emb_results.csv           # 임베딩 비교 결과(샘플)
   ├─ which-vector-db-to-use... # 벡터DB 메모
   ├─ nureongi/                 # RAG 내부 모듈 계층
   │  ├─ loaders.py             # 파일/피드 로더
   │  ├─ indexing.py            # 청크/인덱싱 파이프라인
   │  ├─ vectorstore.py         # Chroma/FAISS 생성/연결
   │  ├─ chain.py               # 리트리버 + 생성 체인
   │  └─ raptor.py              # RAPTOR 요약 트리(옵션)
   ├─ services/                 # 서비스 계층
   │  ├─ rag_service.py         # ★ RAG 쿼리 실행, 입력 정규화/후처리
   │  └─ strategy.py            # LLM/임베딩 백엔드 전략
   ├─ schemas/                  # Pydantic 스키마
   │  ├─ query.py               # 요청 모델
   │  └─ response.py            # 응답 모델
   ├─ news_cat/                 # 도메인 룰/카테고리(실험)
   ├─ observability/            # 로깅/메트릭 확장 포인트
   ├─ dataset/                  # 원천 데이터(로컬)
   ├─ output/                   # 산출물(요약/게시물 등)
   ├─ cache/                    # 임시 캐시
   ├─ .chroma/                  # Chroma 퍼시스턴스 디렉터리
   └─ faiss_index/              # FAISS 인덱스(옵션)
```

> 실제 폴더는 스냅샷에 따라 다를 수 있습니다. 상단 트리는 현재 레포 스크린샷을 기준으로 정리했습니다.

---

## 환경 변수(.env)

```dotenv
# LLM 백엔드
LLM_BACKEND=gemini            # 예시: openai | gemini | ...
GEMINI_MODEL=gemini-2.0-flash # 예시
OPENAI_MODEL=gpt-4o-mini      # 예시

# 임베딩
EMB_BACKEND=hf
EMB_MODEL=BAAI/bge-base-en-v1.5

# 서버
HOST=0.0.0.0
PORT=8001
CORS_ALLOW_ORIGINS=http://localhost:5500

# VectorStore
VECTORSTORE_PROVIDER=chroma   # chroma | faiss
CHROMA_DIR=./src/.chroma
FAISS_DIR=./src/faiss_index

# RAPTOR
RAPTOR_ENABLED=false          # true/false
RAPTOR_TARGET_K=3

# (선택) DB 로깅
DB_URL=sqlite:///./rag_logs.db
```

---

## 실행

### 로컬

```bash
# venv 준비 후
pip install -r requirements.txt

# .env 준비
cp .env.example .env
# 필요한 값 채우기

# 실행
cd src
uvicorn api_rag:app --reload --host 0.0.0.0 --port 8001
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

#### Request (JSON)

```json
{
  "query": "최근 LLM 경량화 동향을 요약해줘.",
  "top_k": 5,
  "fetch_k": 20,
  "lambda_mult": 0.5,
  "days": 60,               // (선택) 최근 N일 메타 필터
  "persona": "auto"         // (선택) 프롬프트 스타일
}
```

> 입력은 내부에서 **문자열 강제 정규화**됩니다. (dict로 들어와 AttributeError가 나던 이슈 방지)

#### Response (JSON)

```json
{
  "answer": "요약 답변 문자열...",
  "refs": [
    {
      "title": "기사/문서 제목",
      "url": "https://example.com/article",
      "score": 0.83,
      "published_at": "2025-08-15T05:29:50Z"
    }
  ],
  "meta": {
    "retriever": { "k": 5, "fetch_k": 20, "lambda_mult": 0.5 },
    "vectorstore": "chroma",
    "elapsed_ms": 412
  }
}
```

> 참조 URL은 `link → url → guid` 우선순위로 추출하여 `refs`에 포함합니다.

#### curl 테스트

```bash
curl -X POST "http://localhost:8001/redfin_target-insight" \
  -H "Content-Type: application/json" \
  -d '{"query":"RAG 최신 기법 핵심만","top_k":5,"fetch_k":20,"lambda_mult":0.5,"days":60}'
```

성공 DoD:

* `200 OK`, `answer` 문자열, `refs`에 실제 URL ≥ 1
* 서버 로그에 `OPTIONS /redfin_target-insight 200` → `POST /redfin_target-insight 200`
* `get_relevant_documents` Deprecation 경고 없음(→ `retriever.invoke` 사용)

---

### 2) (예정) `POST /redfin_news` — 자동 기사 출간

공통 RAG 파이프라인 위에서 **수집 → 요약/정제 → 태깅 → 게시물 생성**까지 자동화합니다.
초기 버전은 초안 생성 후 수동 검수 단계를 포함합니다.

#### 계획 요청/응답 스펙(초안)

```json
// Request
{
  "topic": "AI 정책/규제",
  "days": 3,
  "target_channel": "blog",   // blog | newsletter | web
  "style": "briefing"         // headline | briefing | deep-dive
}

// Response(초안)
{
  "post_id": "draft_20250816_001",
  "title": "이번 주 AI 정책 핵심 5가지",
  "content_md": "## 1. ...",
  "tags": ["policy/Regulation","geo/US"],
  "refs": [{ "title":"...", "url":"..." }]
}
```

---

## 프론트엔드 연계

### CORS

* `.env`의 `CORS_ALLOW_ORIGINS`에 프론트 도메인(예: `http://localhost:5500`)을 등록합니다.
* 프리플라이트(OPTIONS) 200 응답을 보장합니다.

### 일반 요청 (fetch)

```javascript
async function askRag(q) {
  const res = await fetch("http://localhost:8001/redfin_target-insight", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ query: q, top_k: 5, fetch_k: 20, lambda_mult: 0.5 })
  });
  const data = await res.json();
  // data.answer / data.refs 사용
}
```

### 스트리밍(예정)

두 가지 옵션 중 하나를 채택할 예정입니다.

1. **SSE(Server-Sent Events)**

   * 서버: `text/event-stream`으로 토큰 단위 전송
   * 클라이언트:

   ```javascript
   const es = new EventSource("http://localhost:8001/redfin_target-insight/stream?query=...");
   es.onmessage = (e) => { append(e.data); };
   es.onerror = () => es.close();
   ```

2. **Chunked JSON Lines**

   * 서버: `StreamingResponse`로 `{"delta":"..."}\n` 반복 전송
   * 클라이언트: Fetch + ReadableStream으로 라인 파싱

초기 구현은 **SSE**를 권장합니다(브라우저 호환/간결함).

---

## 로깅/관측(설계)

요청-응답/에러/LLM토큰/리트리버 결과를 DB에 저장해 **재현/품질 추적**에 활용합니다.

* DB: SQLite(로컬) → PostgreSQL(배포)로 이관 가능
* 테이블 초안:

```sql
CREATE TABLE api_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  endpoint TEXT,
  status INTEGER,
  query TEXT,
  answer TEXT,
  refs_json TEXT,
  retriever_json TEXT,
  elapsed_ms INTEGER,
  error TEXT
);
```

* Pydantic 모델은 `schemas/response.py`에 정의, `observability/`에 저장 모듈 확장

---

## 프로맵트 엔지니어링 가이드(요약)

* **입력 정규화**: 질문이 dict/복합형으로 들어와도 내부에서 문자열로 강제 변환
* **컨텍스트 구성**: `k/fetch_k/lambda_mult` 기본값은 보수적으로, 실패 시 k 축소
* **안전성**: 금칙어/누설 방지 룰 템플릿화, 역할 프롬프트 최소화
* **참조 링크 품질**: `link → url → guid` 우선순위, 중복 제거/도메인 화이트리스트(선택)

---

## 개발 체크리스트

* [x] `POST /redfin_target-insight` 동작
* [x] CORS(5500→8001) 프리플라이트 200
* [x] Retriever deprecation 대응: `retriever.invoke`
* [ ] (선택) “최근 N일” 메타 필터
* [ ] DB 로깅 저장(요청/응답/refs/메트릭)
* [ ] SSE 스트리밍 응답
* [ ] `POST /redfin_news` 자동 출간(초안 → 검수 → 게시)

---

## 트러블슈팅 메모

* `AttributeError: ... embed_documents`: 입력이 dict로 들어온 사례. **문자열 강제 변환**으로 해결.
* `get_relevant_documents` Deprecation: **`retriever.invoke(query)`** 사용.
* RAPTOR 호출량 제한: `.env`에서 `RAPTOR_ENABLED=false` 또는 `RAPTOR_TARGET_K` 축소.

---

## 라이선스

사내/팀 프로젝트 기준에 따릅니다. 외부 공개 시 라이선스 명시 필요.

---

## 참고

* 임베딩 기본: **BGE-base** (비용/지연/인덱스 크기 균형)
* VectorStore 기본: **Chroma** (로컬 개발 생산성), 확장 시 Qdrant/Weaviate 고려
* 프론트엔드 테스트: `http://localhost:5500/test_rag_client.html` (간단 폼에서 POST)

---
### 업데이트 내역
- 2025-08-27 : (강충원) api_rag.py로 RAG API 기능 완료 