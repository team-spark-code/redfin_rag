# WorkLaborCollector

고용노동부 정책자료 크롤러

Playwright를 활용하여 고용노동부 정책자료 게시판에서 PDF 문서를 수집하는 크롤러입니다.

## 특징

- **상속 구조**: `base_collector.py`의 `Collector` 클래스를 상속하여 기존 기능 활용
- **Playwright 활용**: 동적 콘텐츠와 AJAX 응답을 처리하는 현대적인 웹 크롤링
- **두 가지 크롤링 모드**: 
  - 단일 페이지 재사용 모드 (기본값, 서버 부하 최소화)
  - 여러 컨텍스트 모드 (독립적인 브라우저 컨텍스트 사용)
- **안전한 파일명 처리**: 특수문자 제거 및 파일명 정리
- **진행률 표시**: 다운로드 진행률과 파일 크기 표시
- **ANSI 색상 출력**: 터미널에서 보기 좋은 컬러 출력
- **통계 추적**: 다운로드 성공/실패 통계 및 파일 크기 추적

## 설치

```bash
# Playwright 설치
pip install playwright
playwright install

# 프로젝트 의존성 설치
pip install -r requirements.txt
```

## 기본 사용법

### 1. 기본 크롤링 (단일 페이지 재사용 모드)

```python
import asyncio
from core.ingestion.collector.worklabor_crawler import WorkLaborCollector

async def main():
    # 기본 설정으로 크롤러 생성
    collector = WorkLaborCollector()
    
    # 문서 크롤링 실행 (최대 5개 문서)
    result = await collector.crawl_documents(max_documents=5)
    
    print(f"크롤링 결과: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 여러 컨텍스트 모드 사용

```python
import asyncio
from core.ingestion.collector.worklabor_crawler import WorkLaborCollector

async def main():
    # 여러 컨텍스트 모드로 크롤러 생성
    collector = WorkLaborCollector(
        output_dir="output/worklabor_multiple",
        use_multiple_contexts=True
    )
    
    # 문서 크롤링 실행
    result = await collector.crawl_documents(max_documents=3)
    
    print(f"크롤링 결과: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. 커스텀 출력 디렉토리

```python
import asyncio
from core.ingestion.collector.worklabor_crawler import WorkLaborCollector

async def main():
    # 커스텀 출력 디렉토리 설정
    collector = WorkLaborCollector(
        output_dir="output/custom_worklabor",
        use_multiple_contexts=False
    )
    
    # 문서 크롤링 실행
    result = await collector.crawl_documents(max_documents=2)
    
    print(f"크롤링 결과: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 클래스 구조

### WorkLaborCollector 클래스

`base_collector.py`의 `Collector` 클래스를 상속하여 다음 기능을 제공합니다:

#### 생성자 매개변수

- `output_dir` (str): 다운로드 파일 저장 디렉토리 (기본값: "output/worklabor")
- `use_multiple_contexts` (bool): 여러 컨텍스트 사용 여부 (기본값: False)

#### 주요 메서드

- `crawl_documents(max_documents: int = 5)`: 문서 크롤링 실행
- `start_browser()`: Playwright 브라우저 시작
- `stop_browser()`: Playwright 브라우저 종료
- `get_download_stats()`: 다운로드 통계 반환

#### 상속된 메서드 (Collector 클래스)

- `create_document_info()`: DocumentInfo 객체 생성
- `get_summary()`: 수집기 현황 요약
- `get_collection_status()`: 현재 수집 상태
- `get_download_history()`: 다운로드 기록

## 크롤링 모드

### 1. 단일 페이지 재사용 모드 (기본값)

- 하나의 브라우저 컨텍스트와 페이지를 재사용
- 서버 부하 최소화
- 메모리 사용량 적음
- 상대적으로 느린 처리 속도

### 2. 여러 컨텍스트 모드

- 각 문서마다 새로운 브라우저 컨텍스트 생성
- 독립적인 브라우저 세션
- 더 빠른 처리 속도
- 서버 부하 증가 가능성

## 출력 형식

### 다운로드된 파일

```
output/worklabor/
├── 문서제목_1_1_파일명.pdf
├── 문서제목_2_1_첨부파일명.pdf
├── extracted_links.json
└── worklabor_collector.log
```

### 파일명 정리 규칙

- 개행문자, 탭문자 제거
- "다운로드바로보기" 등 불필요한 텍스트 제거
- 특수문자 제거 (한글, 영문, 숫자, 일부 특수문자만 허용)
- 중복된 .pdf 확장자 제거
- 파일명 길이 255자 제한

### JSON 출력 형식

```json
{
  "success": true,
  "total_links": 5,
  "total_downloaded": 3,
  "download_stats": {
    "total_links": 5,
    "successful_downloads": 3,
    "failed_downloads": 0,
    "total_file_size": 2048576
  },
  "output_dir": "output/worklabor"
}
```

## 터미널 출력

### ANSI 색상 코드

- **헤더**: 보라색 굵은 글씨
- **성공**: 초록색 ✓
- **정보**: 파란색 ℹ
- **경고**: 노란색 ⚠
- **에러**: 빨간색 ✗

### 출력 예시

```
============================================================
                    고용노동부 정책자료 PDF 크롤러
============================================================

ℹ 페이지 제목: 고용노동부 정책자료
ℹ 발견된 링크 수: 5
ℹ 다운로드 시작: 2024년_고용정책_1_1_정책자료.pdf
ℹ 진행률: 10.0% (205KB/2.0MB)
ℹ 진행률: 20.0% (410KB/2.0MB)
✓ 다운로드 완료: 2024년_고용정책_1_1_정책자료.pdf (2.0MB)
```

## 테스트

### 테스트 스크립트 실행

```bash
# 전체 테스트 실행
python test_worklabor_collector.py

# 사용 예시 실행
python example_worklabor_usage.py
```

### 테스트 내용

1. **상속된 메서드 테스트**: Collector 클래스의 메서드들이 제대로 상속되었는지 확인
2. **단일 페이지 모드 테스트**: 기본 크롤링 모드 테스트
3. **여러 컨텍스트 모드 테스트**: 여러 컨텍스트 사용 모드 테스트

## 주의사항

1. **서버 부하**: 과도한 요청으로 인한 서버 부하 방지를 위해 요청 간 1초 대기
2. **브라우저 리소스**: 브라우저 인스턴스는 반드시 종료해야 함
3. **파일 권한**: 출력 디렉토리에 쓰기 권한 필요
4. **네트워크 연결**: 안정적인 인터넷 연결 필요

## 문제 해결

### 일반적인 문제

1. **Playwright 설치 오류**
   ```bash
   playwright install
   ```

2. **브라우저 시작 실패**
   - 시스템에 Chrome 브라우저 설치 확인
   - 방화벽 설정 확인

3. **다운로드 실패**
   - 네트워크 연결 확인
   - 출력 디렉토리 권한 확인
   - 서버 응답 상태 확인

### 로그 확인

```bash
# 크롤러 로그 확인
cat worklabor_collector.log

# 추출된 링크 정보 확인
cat output/worklabor/extracted_links.json
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 