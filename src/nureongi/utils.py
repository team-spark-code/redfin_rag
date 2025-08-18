from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
from langchain.schema import Document
import json

def find_pdf_files(root: Path, pattern: str = "**/*.pdf") -> List[Path]:
    return [p for p in root.rglob(pattern) if p.is_file()]

def serialize_document(doc: Document) -> Dict[str, Any]:
    """Langchain Document -> JSON 직렬화용 dict"""
    meta = dict(doc.metadata or {})

    # 경로/페이지 등 핵심 메타는 보존
    keep = {k: meta.get(k) for k in ["source", "file_path", "page", "total_pages"] if k in meta}

    # 그 외 메타도 함께 기록
    for k, v in meta.items():
        if k not in keep:
            keep[k] = v
    return {
        "page_content": doc.page_content,
        "metadata": keep
    }

def save_jsonl(path: Path, docs: Iterable[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(serialize_document(d), ensure_ascii=False) + "\n")



