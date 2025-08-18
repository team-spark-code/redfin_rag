# Ingestion 파이프라인

고용노동부 PDF 문서를 처리하는 통합 파이프라인입니다.

## 구조

```
core/ingestion/
├── text_extractor.py          # 텍스트 추출 모듈
├── text_processor.py          # 텍스트 전처리 모듈
├── pipeline/
│   └── worklabor_pipeline.py  # 고용노동부 문서 처리 파이프라인
└── loaders/                   # 기존 로더들
    ├── base_loader.py
    ├── pdf_loader.py
    └── ...
```

## 모듈별 설명

### 1. TextExtractor (`text_extractor.py`)

PDF 파일에서 텍스트를 추출하는 모듈입니다.

#### 주요 클래스

- **TextExtractor**: 텍스트 추출 기본 클래스
- **PDFTextExtractor**: PDF 파일 전용 텍스트 추출기
- **WorkLaborPDFExtractor**: 고용노동부 PDF 파일 전용 추출기

#### 기능

- PDF 파일에서 텍스트 추출
- 페이지별 메타데이터 관리
- 파일명에서 문서 정보 추출
- JSON 및 TXT 형태로 저장

#### 사용 예시

```python
from core.ingestion.text_extractor import WorkLaborPDFExtractor

# 고용노동부 PDF 텍스트 추출기 생성
extractor = WorkLaborPDFExtractor(output_dir="output/extracted_text/worklabor")

# PDF 파일에서 텍스트 추출
pdf_file = Path("output/worklabor/document.pdf")
file_info = extractor.extract_text(pdf_file)

# 추출된 텍스트 저장
extractor.save_extracted_text(file_info)
```

### 2. TextProcessor (`text_processor.py`)

추출된 텍스트를 전처리하는 모듈입니다.

#### 주요 클래스

- **TextProcessor**: 텍스트 전처리 기본 클래스
- **TextChunker**: 텍스트 청킹 클래스
- **WorkLaborTextProcessor**: 고용노동부 문서 전용 전처리기

#### 기능

- 텍스트 정제 및 정규화
- 특수문자 처리
- 고용노동부 문서 특화 정제
- 텍스트 청킹 (500자 단위, 50자 오버랩)

#### 사용 예시

```python
from core.ingestion.text_processor import WorkLaborTextProcessor, TextChunker

# 고용노동부 텍스트 전처리기 생성
processor = WorkLaborTextProcessor(output_dir="output/processed_text/worklabor")

# 문서 전처리
processed_info = processor.process_worklabor_document(file_info)

# 전처리된 문서 저장
processor.save_processed_document(processed_info)

# 텍스트 청킹
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.split_text(text, metadata)
```

### 3. WorkLaborPipeline (`pipeline/worklabor_pipeline.py`)

전체 파이프라인을 관리하는 통합 클래스입니다.

#### 파이프라인 단계

1. **텍스트 추출 단계**: PDF 파일에서 텍스트 추출
2. **텍스트 전처리 단계**: 추출된 텍스트 정제 및 정규화
3. **텍스트 청킹 단계**: 전처리된 텍스트를 청크로 분할

#### 출력 디렉토리 구조

```
output/
├── worklabor/                    # 원본 PDF 파일
├── extracted_text/worklabor/     # 추출된 텍스트
├── processed_text/worklabor/     # 전처리된 텍스트
└── chunked_text/worklabor/      # 청킹된 텍스트
```

#### 사용 예시

```python
from core.ingestion.pipeline.worklabor_pipeline import WorkLaborPipeline

# 파이프라인 생성
pipeline = WorkLaborPipeline(
    input_dir="output/worklabor",
    extracted_dir="output/extracted_text/worklabor",
    processed_dir="output/processed_text/worklabor",
    chunked_dir="output/chunked_text/worklabor"
)

# 전체 파이프라인 실행
summary = pipeline.run_pipeline()
```

## 실행 방법

### 1. 전체 파이프라인 실행

```bash
python run_worklabor_pipeline.py
```

### 2. 테스트용 파이프라인 실행

```bash
python run_worklabor_pipeline.py
# 선택: 2 (테스트용 파이프라인)
```

### 3. 개별 모듈 테스트

```bash
python test_pdf_extraction.py
```

## 출력 파일 형식

### 1. 추출된 텍스트 (JSON)

```json
{
  "file_name": "document.pdf",
  "file_path": "output/worklabor/document.pdf",
  "file_size": 1048576,
  "extraction_date": "2025-01-15T10:30:00",
  "total_pages": 10,
  "pages": [
    {
      "page_number": 1,
      "text_content": "페이지 내용...",
      "text_length": 1500,
      "metadata": {...}
    }
  ],
  "source": "고용노동부",
  "source_detail": "고용노동부 정책자료",
  "document_info": {
    "document_title": "문서 제목",
    "sequence_number": 1,
    "attachment_number": 1
  }
}
```

### 2. 전처리된 텍스트 (JSON)

```json
{
  "file_name": "document.pdf",
  "processing_date": "2025-01-15T10:35:00",
  "total_pages": 10,
  "total_original_length": 15000,
  "total_cleaned_length": 14500,
  "processed_pages": [
    {
      "page_number": 1,
      "original_text": "원본 텍스트...",
      "cleaned_text": "정제된 텍스트...",
      "original_length": 1500,
      "cleaned_length": 1450
    }
  ],
  "source_info": {
    "source": "고용노동부",
    "source_detail": "고용노동부 정책자료",
    "processor_type": "WorkLaborTextProcessor"
  }
}
```

### 3. 청킹된 텍스트 (JSON)

```json
{
  "file_name": "document.pdf",
  "total_chunks": 25,
  "chunking_date": "2025-01-15T10:40:00",
  "chunks": [
    {
      "chunk_id": "document.pdf_chunk_1",
      "chunk_number": 1,
      "text_content": "청크 내용...",
      "text_length": 500,
      "metadata": {
        "file_name": "document.pdf",
        "source": "고용노동부",
        "document_info": {...}
      }
    }
  ]
}
```

## 통계 정보

파이프라인 실행 후 다음과 같은 통계 정보를 제공합니다:

- 총 파일 수
- 추출 성공/실패 수
- 전처리 성공/실패 수
- 청킹 성공/실패 수
- 총 페이지 수
- 총 청크 수
- 총 텍스트 길이

## 로그 파일

각 모듈별로 로그 파일이 생성됩니다:

- `text_extractor.log`: 텍스트 추출 로그
- `text_processor.log`: 텍스트 전처리 로그
- `worklabor_pipeline.log`: 파이프라인 실행 로그

## 주의사항

1. **메모리 사용량**: 대용량 PDF 파일 처리 시 메모리 사용량에 주의
2. **처리 시간**: 파일 크기에 따라 처리 시간이 달라질 수 있음
3. **디스크 공간**: 추출된 텍스트가 원본 PDF보다 클 수 있음
4. **파일 권한**: 출력 디렉토리에 쓰기 권한 필요

## 확장 가능성

이 파이프라인은 다음과 같이 확장할 수 있습니다:

1. **다른 문서 형식 지원**: DOCX, HWP 등
2. **다른 정부 기관 지원**: 국세청, 행정안전부 등
3. **고급 전처리**: OCR, 이미지 텍스트 추출 등
4. **벡터 데이터베이스 연동**: FAISS, Chroma 등

## 문제 해결

### 일반적인 문제

1. **PDF 파일 읽기 실패**
   - PyMuPDF 설치 확인: `pip install pymupdf`
   - 파일 손상 여부 확인

2. **메모리 부족**
   - 파일 크기 확인
   - 배치 처리로 변경

3. **텍스트 추출 실패**
   - 스캔된 PDF인지 확인
   - OCR 필요 여부 확인

### 로그 확인

```bash
# 로그 파일 확인
tail -f text_extractor.log
tail -f text_processor.log
tail -f worklabor_pipeline.log
``` 