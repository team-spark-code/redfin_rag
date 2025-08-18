# 누렁이 - 국세청 문서 크롤러 가이드

## 개요

대한민국 정부 공식 문서 RAG 기반 웹에디터 '누렁이' 프로젝트의 국세청 문서 자동 수집 도구입니다.

국세청 게시판에서 새로운 문서가 업로드되는지 하루 단위로 확인하고, 새로운 문서가 있으면 자동으로 다운로드하여 저장 및 관리합니다.

## 주요 기능

- 🔍 국세청 게시판 자동 모니터링
- 📄 새 문서 자동 감지 및 다운로드
- 📊 다운로드 기록 및 상태 관리
- 🤖 스케줄링 및 데몬 모드 지원
- 🔔 알림 기능 (텔레그램)

## 지원 게시판

| 게시판 ID | 설명 | URL |
|-----------|------|-----|
| `tax_law` | 개정세법해설 (기본값) | [링크](https://s.nts.go.kr/nts/na/ntt/selectNttList.do?mi=7133&bbsId=1083) |
| `income_tax` | 종합소득세 | [링크](https://s.nts.go.kr/nts/na/ntt/selectNttList.do?mi=7133&bbsId=1084) |
| `corporate_tax` | 법인세 | [링크](https://s.nts.go.kr/nts/na/ntt/selectNttList.do?mi=7133&bbsId=1085) |

## 설치

### 1. Python 환경 설정

```bash
# Python 3.10+ 필요
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install playwright aiohttp aiofiles
playwright install chromium
```

### 2. 프로젝트 설정

```bash
# 프로젝트 루트에서
cd rag  # 프로젝트 디렉토리로 이동
```

## 사용법

### CLI 사용

```bash
# 1회 실행 (테스트)
python scripts/nts_cli.py run

# 특정 게시판 크롤링
python scripts/nts_cli.py run --board income_tax

# 연결 테스트
python scripts/nts_cli.py test

# 일일 모니터링 시작 (24시간마다)
python scripts/nts_cli.py monitor

# 6시간마다 체크
python scripts/nts_cli.py monitor --interval 6

# 상태 확인
python scripts/nts_cli.py status
```

### Python 코드에서 사용

```python
import asyncio
from core.ingestion.collector.nts_crawler import NTSCrawler

async def example():
    # 크롤러 생성
    crawler = NTSCrawler(board_type="tax_law", output_dir="downloads/nts")
    
    # 1회 실행
    result = await crawler.run_once()
    print(f"새 문서: {result['new_documents']}개")
    
    # 일일 모니터링 시작
    # await crawler.run_daily_check(interval_hours=24)

asyncio.run(example())
```

### 편의 함수 사용

```python
import asyncio
from core.ingestion.collector.nts_crawler import crawl_nts_once

# 간단한 1회 크롤링
result = asyncio.run(crawl_nts_once("tax_law", "downloads/nts"))
print(result)
```

## 출력 구조

```
output/nts/
├── download_history.json    # 다운로드 기록
├── collector.log           # 로그 파일
└── 문서명_20241201_143022.pdf  # 다운로드된 파일들
```

## 자동화 설정

### 1. Cron 설정 (Linux/macOS)

```bash
# crontab 편집
crontab -e

# 매일 오전 9시에 실행
0 9 * * * cd /path/to/rag && python scripts/nts_cli.py run

# 6시간마다 실행
0 */6 * * * cd /path/to/rag && python scripts/nts_cli.py run
```

### 2. systemd 서비스 (Linux)

```ini
# /etc/systemd/system/nts-crawler.service
[Unit]
Description=NTS Document Crawler
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/rag
ExecStart=/path/to/venv/bin/python scripts/nts_cli.py monitor --interval 24
Restart=always
RestartSec=3600
Environment=PYTHONPATH=/path/to/rag

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 등록 및 시작
sudo systemctl enable nts-crawler.service
sudo systemctl start nts-crawler.service
sudo systemctl status nts-crawler.service
```

### 3. Docker 사용

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Playwright 의존성 설치
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt && playwright install chromium

# 프로젝트 복사
COPY . .

# 실행
CMD ["python", "scripts/nts_cli.py", "monitor", "--interval", "24"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  nts-crawler:
    build: .
    volumes:
      - ./output:/app/output
    environment:
      - TZ=Asia/Seoul
      - PYTHONPATH=/app
    restart: unless-stopped
```

### 4. Windows 작업 스케줄러

1. 작업 스케줄러 열기
2. 기본 작업 만들기
3. 트리거: 매일
4. 동작: 프로그램 시작
   - 프로그램: `python.exe`
   - 인수: `scripts/nts_cli.py run`
   - 시작 위치: `C:\path\to\rag`

## 알림 설정

### 텔레그램 알림

```python
from core.ingestion.collector.nts_crawler import NTSCrawler

crawler = NTSCrawler()
crawler.set_telegram_notification(
    bot_token="YOUR_BOT_TOKEN",
    chat_id="YOUR_CHAT_ID"
)

# 테스트 알림
await crawler.test_notification()
```

## 로깅

로그는 다음 위치에 저장됩니다:
- 파일: `{output_dir}/collector.log`
- 콘솔: 실시간 출력

로그 레벨은 환경변수로 제어할 수 있습니다:
```bash
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## 문제 해결

### 일반적인 문제

1. **브라우저 실행 실패**
   ```bash
   playwright install chromium
   ```

2. **권한 오류**
   ```bash
   chmod +x scripts/nts_cli.py
   ```

3. **네트워크 타임아웃**
   - VPN 연결 확인
   - 방화벽 설정 확인

4. **메모리 부족**
   - Playwright 브라우저가 많은 메모리 사용
   - 동시 실행 수 줄이기

### 디버그 모드

```bash
# 디버그 로그 활성화
export LOG_LEVEL=DEBUG
python scripts/nts_cli.py run

# 브라우저 헤드리스 모드 비활성화 (개발용)
# crawler.py에서 headless=False로 설정
```

## API 참조

### NTSCrawler 클래스

```python
class NTSCrawler(Collector):
    def __init__(self, board_type: str = "tax_law", output_dir: str = "output/nts")
    async def discover_documents(self) -> List[DocumentInfo]
    async def collect_new_documents(self) -> Dict[str, Any]
    async def run_once(self) -> Dict[str, Any]
    async def run_daily_check(self, interval_hours: int = 24)
```

### 편의 함수

```python
async def crawl_nts_once(board_type: str, output_dir: str) -> Dict[str, Any]
async def start_nts_daily_monitor(board_type: str, output_dir: str, interval_hours: int)
```

## 라이선스

MIT License

## 지원

이슈나 문의사항은 GitHub Issues에 등록해주세요. 