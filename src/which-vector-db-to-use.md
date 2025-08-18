### 환경별 벡터 DB 전략
A. 정부 시스템(폐쇄망/보안최우선)
요구: 퍼블릭 클라우드 접속 금지, 전송구간 TLS/mTLS 강제, 감사추적/접근통제, 오프라인 백업, 장기 유지보수
권장:
우선순위: Milvus(온프레/TLS) → Qdrant(온프레/TLS) → FAISS(on-disk)
배포: 에어갭/내부 K8s, PV로 SSD/NVMe, 노드 라벨링+anti-affinity, offline 레지스트리
보안: grpcs/https만 허용, 방화벽로 19530/6333 제한, mTLS(가능 시), RBAC, 감사 로그 중앙집중
인덱스: HNSW 또는 IVF_FLAT(질의패턴에 따라), 차원/거리함수는 임베딩 모델 고정 후 컬렉션 스키마 고정
백업: Milvus 백업/로드, Qdrant 스냅샷 주기화 + 원장형 백업, DR용 오프사이트 보관
운영: 리소스 요청/리밋, 모니터링(Prometheus/Grafana), 스키마 변경은 신규 컬렉션 롤링
B. 중소기업(비용/운영밸런스)
요구: 빠른 시간-투-밸류, 인력 한정, 예측 가능한 비용
권장:
우선순위: Qdrant Cloud(https+API Key) → Milvus(매니지드/자가호스팅) → FAISS(on-disk)
배포: 초기엔 Qdrant Cloud로 시작(운영 단순화) → 트래픽/비용临계 시 Milvus 자가호스팅 전환
보안: VPC 피어링/전용 엔드포인트, 키 롤테이션, 최소권한 키
백업: Qdrant 스냅샷(+클라우드 백업), Milvus PVC 스냅샷/주기적 dump
운영: 자동 스케일 기준치 정의(QPS/latency/메모리 사용률), 스테이징 분리
C. 개인(로컬/단일 노드)
요구: 설치 간단, 비용=0~저렴, 노트북/데스크톱
권장:
우선순위: 로컬 Qdrant(Docker) → FAISS(on-disk)
배포: 도커 1컨테이너 또는 완전 로컬 FAISS
백업: FAISS 인덱스 디렉터리 압축 보관, 주기적 저장
운영: 임베딩/차원 고정 후 인덱스 재생성 최소화, 벡터 수↑ 시 HNSW/IVF 고려