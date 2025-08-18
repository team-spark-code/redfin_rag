from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from typing import List

def load_single_pdf(path: Path, max_pages: Optional[int] = None) -> List[Document]:
    """
    개별 PDF 로드.
    - PyMuPDFLoader 사용
    - max_pages 지정 시 앞쪽 페이지만 로드(긴 문서 샘플링/가속)
    """
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    # 메타데이터 정규화
    for d in docs:
        d.metadata.setdefault("file_path", str(path))
        d.metadata.setdefault("source", str(path))
    if max_pages is not None and max_pages > 0:
        docs = [d for d in docs if int(d.metadata.get("page", 0)) < max_pages]
    return docs
