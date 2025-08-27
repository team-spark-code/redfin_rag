# nureongi/cache.py
import pickle
from pathlib import Path
from typing import Optional, Dict, List
from langchain_core.documents import Document

def load_cache(cache_path: Optional[Path]) -> Dict[str, List[Document]]:
    if not cache_path or not cache_path.exists():
        return {}
    with cache_path.open("rb") as f:
        return pickle.load(f)

def save_cache(cache_path: Optional[Path], cache: Dict[str, List[Document]]) -> None:
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

