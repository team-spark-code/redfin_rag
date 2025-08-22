# nureongi/raptor.py
from __future__ import annotations
from typing import List, Optional
import math
import numpy as np

try:
    import umap
    from sklearn.cluster import AgglomerativeClustering
except Exception as e:
    raise ImportError(
        "RAPTOR 의존성이 없습니다. `pip install umap-learn scikit-learn numpy` 후 다시 시도하세요."
    ) from e

from langchain_core.documents import Document
from .vectorstore import build_embedder


def _estimate_n_clusters(n_docs: int) -> int:
    # 문서 수가 적으면 그대로 반환
    if n_docs < 20:
        return max(1, n_docs)
    # √N 휴리스틱, 2~50 사이로 클램프
    k = int(round(math.sqrt(n_docs)))
    return max(2, min(50, k))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def raptor_reduce(
    docs: List[Document],
    n_neighbors: Optional[int] = None,
    n_components: int = 10,
    n_clusters: Optional[int] = None,
    limit_reps: Optional[int] = 200,
) -> List[Document]:
    """
    UMAP + HAC로 군집 대표 문서를 선택해 축약 반환.
    - 입력: 원본 Document 리스트(요약문 중심; 제목 누수 없음 가정)
    - 출력: 대표 문서들(원본 Document; payload/metadata 보존)

    Args:
        n_neighbors: UMAP 이웃 수. None이면 min(50, N-1).
        n_components: UMAP 차원 수.
        n_clusters: 군집 수. None이면 √N 휴리스틱.
        limit_reps: 대표 문서 상한. None이면 제한 없음.
    """
    N = len(docs)
    if N == 0:
        return []
    if N < 8:
        # 데이터 적으면 그대로 반환
        return docs

    # 1) 임베딩
    emb = build_embedder()
    X = emb.embed_documents([d.page_content for d in docs])  # List[List[float]]
    X = np.asarray(X, dtype=np.float32)

    # 2) 차원 축소 (UMAP)
    n_neighbors = n_neighbors or max(5, min(50, N - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    Z = reducer.fit_transform(X)  # (N, n_components)

    # 3) 클러스터링 (HAC, 유클리드 공간에서 동작)
    n_clusters = n_clusters or _estimate_n_clusters(N)
    n_clusters = max(1, min(n_clusters, N))
    hac = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")
    labels = hac.fit_predict(Z)  # (N,)

    # 4) 클러스터 대표 선택(centroid에 가장 가까운 문서)
    reps: List[Document] = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        # 클러스터 중심(임베딩 공간에서 평균)
        centroid = X[idx].mean(axis=0)
        # 코사인 유사도 기준 대표 선택
        best_i = None
        best_sim = -1.0
        for i in idx:
            sim = _cosine_sim(X[i], centroid)
            if sim > best_sim:
                best_sim = sim
                best_i = i
        if best_i is not None:
            reps.append(docs[int(best_i)])

    # 5) 상한 적용
    if limit_reps is not None and len(reps) > limit_reps:
        reps = reps[:limit_reps]

    return reps
