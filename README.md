# RedFin RAG - 정부 문서 기반 RAG 시스템

정부 문서 기반 RAG(Retrieval-Augmented Generation) 시스템으로, PDF 문서로부터 정보를 추출하고 임베딩 및 벡터 데이터베이스를 활용하여 근거 기반 질의응답 기능을 제공합니다.

## 🚀 새로운 아키텍처 (src/ 디렉토리)

```
redfin_rag/
├── src/                          # 핵심 RAG 시스템
│   ├── main.py                   # 메인 실행 파일
│   ├── nureongi/                 # Nureongi RAG 패키지
│   │   ├── __init__.py           # 패키지 초기화
│   │   ├── vectorstore.py        # 벡터스토어 자동 폴백
│   │   ├── chain.py              # RAG 체인 빌더
│   │   ├── router.py             # 프롬프트 라우터
│   │   ├── persona.py            # 페르소나 기반 프롬프트 관리
│   │   ├── format.py             # 컨텍스트 포맷터
│   │   ├── loaders.py            # PDF 로더
│   │   ├── caches.py             # 캐싱 시스템
│   │   └── utils.py              # 유틸리티 함수
│   ├── how-to-deal-with-prompts.md # 프롬프트 관리 가이드
│   ├── which-vector-db-to-use.md # 벡터 DB 선택 가이드
│   └── note-2025-08-15.md        # 개발 노트
├── dataset/                      # 원본 PDF 문서
├── faiss_index/                  # FAISS 벡터 인덱스
├── output/                       # 테스트 결과 및 출력
├── scripts/                      # 유틸리티 스크립트
│   ├── nts_cli.py               # 국세청 크롤러 CLI
│   └── test_nts_crawler.py      # 크롤러 테스트
├── docs/                         # 문서화
├── logs/                         # 로그 디렉토리
├── requirements.txt              # 의존성
├── env.example                   # 환경변수 예시
├── .env                          # 환경변수 (사용자 생성)
├── .gitignore                    # Git 무시 파일
├── Dockerfile                    # Docker 설정
└── README.md                     # 프로젝트 설명
```

## 🎭 페르소나 시스템

### 7가지 역할별 맞춤 응답

1. **정책입안자** (`gov-policy`): 정부문서/법령 중심 분석
2. **학술연구** (`acad-research`): 통계/선행연구 중심 분석
3. **산업동향** (`industry-analyst`): 시장/기술 동향 분석
4. **예비창업** (`startup-pre`): 규제/지원 정책 안내
5. **Executive Brief** (`exec-brief`): 전략적 요약
6. **실무보고** (`staff-report`): 데이터/사례 중심 보고
7. **대학생 리포트** (`student-ug`): 쉬운 설명과 용어 정의

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd redfin_rag

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성
cp env.example .env

# OpenAI API 키 설정
OPENAI_API_KEY=your_api_key_here
```

### 3. 문서 준비
```bash
# dataset/ 폴더에 분석할 PDF 파일을 배치
cp your_documents.pdf dataset/
```

### 4. 실행
```bash
# RAG 시스템 실행
cd src
python main.py
```

## 📊 성능 최적화

### 병렬 처리
- **PDF 로딩**: 8개 워커로 병렬 처리
- **임베딩 생성**: 배치 처리 (100개씩)
- **예외 처리**: 깨진 PDF 자동 스킵

### 캐싱 시스템
- **PDF 캐시**: pickle 기반 영속화
- **임베딩 캐시**: 재계산 방지
- **검색 결과**: MMR로 다양성 확보

### 벡터 검색 최적화
- **MMR 검색**: k=8, fetch_k=60, lambda_mult=0.25
- **자동 폴백**: Qdrant → FAISS
- **메타데이터 필터링**: 출처 기반 필터링

## 🔧 개발 가이드

### 새로운 페르소나 추가
```python
# src/nureongi/persona.py에 추가
class PersonaSlug(Enum):
    NEW_PERSONA = "new-persona"

# 프롬프트 등록
PROMPT_REGISTRY[PersonaSlug.NEW_PERSONA] = PromptTemplate.from_template("...")
```

### 벡터스토어 확장
```python
# src/nureongi/vectorstore.py에 새로운 백엔드 추가
def auto_qdrant_faiss_milvus(...):
    # Milvus 지원 추가
    pass
```

## 📚 학습 자료

### 주요 가이드
- **프롬프트 관리**: `src/how-to-deal-with-prompts.md`
- **벡터 DB 선택**: `src/which-vector-db-to-use.md`
- **개발 노트**: `src/note-2025-08-15.md`

## 📈 모니터링

### LangSmith 통합
- **실행 추적**: 모든 RAG 체인 실행 로깅
- **성능 모니터링**: 응답 시간, 토큰 사용량
- **품질 평가**: RAGAS 메트릭 자동 계산

### 로깅
```python
# 메타 로깅 예시
client.create_run(
    name="document-summary-run",
    inputs={"query": query},
    outputs={"answer": result[:1000]},
    tags=["RAG", "summary", "frontier-lab"],
)
```

## 📈 Docker 실행

```bash
# Docker 이미지 빌드
docker build -t redfin-rag .

# 컨테이너 실행
docker run -p 8000:8000 redfin-rag
```

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

