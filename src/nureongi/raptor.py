# # nureongi/raptor.py
# from __future__ import annotations
# from typing import List, Optional
# import math
# import numpy as np

# try:
#     import umap
#     from sklearn.cluster import AgglomerativeClustering
# except Exception as e:
#     raise ImportError(
#         "RAPTOR 의존성이 없습니다. `pip install umap-learn scikit-learn numpy` 후 다시 시도하세요."
#     ) from e

# from langchain_core.documents import Document
# from .vectorstore import build_embedder


# def _estimate_n_clusters(n_docs: int) -> int:
#     # 문서 수가 적으면 그대로 반환
#     if n_docs < 20:
#         return max(1, n_docs)
#     # √N 휴리스틱, 2~50 사이로 클램프
#     k = int(round(math.sqrt(n_docs)))
#     return max(2, min(50, k))


# def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#     na = np.linalg.norm(a) + 1e-9
#     nb = np.linalg.norm(b) + 1e-9
#     return float(np.dot(a, b) / (na * nb))


# def raptor_reduce(
#     docs: List[Document],
#     n_neighbors: Optional[int] = None,
#     n_components: int = 10,
#     n_clusters: Optional[int] = None,
#     limit_reps: Optional[int] = 200,
# ) -> List[Document]:
#     """
#     UMAP + HAC로 군집 대표 문서를 선택해 축약 반환.
#     - 입력: 원본 Document 리스트(요약문 중심; 제목 누수 없음 가정)
#     - 출력: 대표 문서들(원본 Document; payload/metadata 보존)

#     Args:
#         n_neighbors: UMAP 이웃 수. None이면 min(50, N-1).
#         n_components: UMAP 차원 수.
#         n_clusters: 군집 수. None이면 √N 휴리스틱.
#         limit_reps: 대표 문서 상한. None이면 제한 없음.
#     """
#     N = len(docs)
#     if N == 0:
#         return []
#     if N < 8:
#         # 데이터 적으면 그대로 반환
#         return docs

#     # 1) 임베딩
#     emb = build_embedder()
#     X = emb.embed_documents([d.page_content for d in docs])  # List[List[float]]
#     X = np.asarray(X, dtype=np.float32)

#     # 2) 차원 축소 (UMAP)
#     n_neighbors = n_neighbors or max(5, min(50, N - 1))
#     reducer = umap.UMAP(
#         n_neighbors=n_neighbors,
#         n_components=n_components,
#         metric="cosine",
#         random_state=42,
#         verbose=False,
#     )
#     Z = reducer.fit_transform(X)  # (N, n_components)

#     # 3) 클러스터링 (HAC, 유클리드 공간에서 동작)
#     n_clusters = n_clusters or _estimate_n_clusters(N)
#     n_clusters = max(1, min(n_clusters, N))
#     hac = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")
#     labels = hac.fit_predict(Z)  # (N,)

#     # 4) 클러스터 대표 선택(centroid에 가장 가까운 문서)
#     reps: List[Document] = []
#     for c in range(n_clusters):
#         idx = np.where(labels == c)[0]
#         if idx.size == 0:
#             continue
#         # 클러스터 중심(임베딩 공간에서 평균)
#         centroid = X[idx].mean(axis=0)
#         # 코사인 유사도 기준 대표 선택
#         best_i = None
#         best_sim = -1.0
#         for i in idx:
#             sim = _cosine_sim(X[i], centroid)
#             if sim > best_sim:
#                 best_sim = sim
#                 best_i = i
#         if best_i is not None:
#             reps.append(docs[int(best_i)])

#     # 5) 상한 적용
#     if limit_reps is not None and len(reps) > limit_reps:
#         reps = reps[:limit_reps]

#     return reps

# def raptor_tree_reduce(
#     docs: List[Document],
#     levels: int = 1,
#     per_level_limit: Optional[int] = 200,
#     n_neighbors: Optional[int] = None,
#     n_components: int = 10,
#     n_clusters: Optional[int] = None,
# ) -> List[Document]:
#     """
#     raptor_reduce(1단계 축약)를 levels 번 반복 적용.
#     - levels <= 1 이면 1단계만 수행.
#     - per_level_limit: 매 레벨 대표 상한(속도/품질 밸런싱)
#     - n_neighbors/n_components/n_clusters: 각 레벨에 동일 적용
#     """
#     cur = docs
#     if not cur or levels <= 0:
#         return cur

#     for _ in range(max(1, levels)):
#         nxt = raptor_reduce(
#             cur,
#             n_neighbors=n_neighbors,
#             n_components=n_components,
#             n_clusters=n_clusters,
#             limit_reps=per_level_limit,
#         )
#         # 수렴 실패 방지: 변화 없으면 중단
#         if not nxt or len(nxt) >= len(cur):
#             break
#         cur = nxt
#     return cur
# nureongi/raptor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os, math, numpy as np

try:
    import umap
    from sklearn.cluster import AgglomerativeClustering
except Exception as e:
    raise ImportError(
        "RAPTOR 의존성이 없습니다. `pip install umap-learn scikit-learn numpy scikit-learn` 후 다시 시도하세요."
    ) from e

from langchain_core.documents import Document
from .vectorstore import build_embedder


# ----------------- 하이퍼파라미터 -----------------
@dataclass(frozen=True)
class RaptorParams:
    max_levels: int = int(os.getenv("RAPTOR_MAX_LEVELS", "3"))
    target_k: int = int(os.getenv("RAPTOR_TARGET_K", "24"))
    per_cluster_topm: int = int(os.getenv("RAPTOR_PER_CLUSTER_TOPM", "6"))
    n_neighbors: Optional[int] = (
        int(os.getenv("RAPTOR_N_NEIGHBORS")) if os.getenv("RAPTOR_N_NEIGHBORS") else None
    )
    n_components: int = int(os.getenv("RAPTOR_N_COMPONENTS", "10"))
    summary_maxtok: int = int(os.getenv("RAPTOR_SUMMARY_MAXTOK", "256"))


# ----------------- 유틸 -----------------
def _estimate_n_clusters(n_docs: int) -> int:
    if n_docs < 20:
        return max(1, n_docs)
    k = int(round(math.sqrt(n_docs)))
    return max(2, min(50, k))

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


# ----------------- 1단계: 군집 + 대표선정 -----------------
def _cluster_and_pick(
    X: np.ndarray,
    docs: List[Document],
    n_neighbors: Optional[int],
    n_components: int,
    n_clusters: Optional[int] = None,
    per_cluster_topm: int = 6,
) -> Tuple[List[List[int]], List[int], Dict[int, List[int]]]:
    """
    UMAP 차축소 → HAC 군집 → 각 클러스터에서 대표 1개 + 상위 M개 인덱스 반환
    Returns:
        clusters: 각 클러스터의 문서 인덱스 리스트
        reps: 각 클러스터 대표 문서 인덱스 리스트
        tops: {cluster_idx: [상위 M개 문서 인덱스]}
    """
    N = X.shape[0]
    n_neighbors = n_neighbors or max(5, min(50, N - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    Z = reducer.fit_transform(X)  # (N, n_components)

    n_clusters = n_clusters or _estimate_n_clusters(N)
    n_clusters = max(1, min(n_clusters, N))
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")  # euclidean 내부 가정
    labels = hac.fit_predict(Z)  # (N,)

    clusters: List[List[int]] = []
    reps: List[int] = []
    tops: Dict[int, List[int]] = {}

    for c in range(n_clusters):
        idx = np.where(labels == c)[0].tolist()
        if not idx:
            continue
        # 대표 = 임베딩 공간 평균에 가장 가까운(코사인)
        centroid = X[idx].mean(axis=0)
        best_i, best_sim = None, -1.0
        for i in idx:
            sim = _cosine_sim(X[i], centroid)
            if sim > best_sim:
                best_sim, best_i = sim, i
        reps.append(int(best_i))  # type: ignore

        # 상위 M개(코사인 기준 내림차순)
        scored = sorted(idx, key=lambda i: _cosine_sim(X[i], centroid), reverse=True)
        tops[c] = scored[: max(1, per_cluster_topm)]
        clusters.append(idx)

    return clusters, reps, tops


# ----------------- 2단계: 클러스터 요약(Document 생성) -----------------
_SUMMARY_TEMPLATE = """다음 자료를 {maxtok} 토큰 이내로 요약하라.
규칙: 사실 왜곡 금지, 수치·날짜·명칭 보존, 중복 제거, 한국어로 작성.
입력:
{items}

출력: 핵심만 간결히.
"""

def _concat_items(docs: List[Document], idxs: List[int], max_chars: int = 8000) -> str:
    parts, n = [], 0
    for i in idxs:
        d = docs[i]
        md = d.metadata or {}
        title = md.get("title") or md.get("headline") or ""
        src = md.get("source") or md.get("site") or md.get("publisher") or ""
        piece = f"- [{src}] {title}\n{d.page_content.strip()}\n"
        if n + len(piece) > max_chars:
            break
        parts.append(piece)
        n += len(piece)
    return "\n".join(parts)

def _summarize_cluster(
    llm, docs: List[Document], idxs: List[int], params: RaptorParams, level: int, cluster_id: int
) -> Document:
    prompt = _SUMMARY_TEMPLATE.format(
        maxtok=params.summary_maxtok,
        items=_concat_items(docs, idxs),
    )
    text = llm.invoke(prompt)  # langchain chat model의 .invoke(str) 가정
    # 메타데이터에 트리 정보 기록
    meta = {
        "level": level,
        "cluster_id": cluster_id,
        "children": idxs,
        "type": "summary",
    }
    return Document(page_content=str(text), metadata=meta)


# ----------------- 3단계: RAPTOR 트리 빌드/압축 -----------------
def raptor_build_and_compress(
    docs: List[Document],
    llm,
    *,
    params: Optional[RaptorParams] = None,
) -> List[Document]:
    """
    재귀적 트리 구성 + 레벨 요약 생성 → 목표 개수(target_k) 이하의 상위 레벨 요약 노드 집합 반환.
    - 입력: 원문 Document들(leaf)
    - 출력: 상위 레벨의 요약 Document들(프롬프트에 바로 투입 가능)
    """
    if not docs:
        return []

    params = params or RaptorParams()

    # 0) 임베딩 캐시(최초 leaf에 대해 한 번만)
    emb = build_embedder()
    X0 = np.asarray(emb.embed_documents([d.page_content for d in docs]), dtype=np.float32)

    cur_docs = docs
    cur_X = X0
    level = 0

    while True:
        level += 1
        # 클러스터링 & 대표/톱M 선택
        clusters, reps, tops = _cluster_and_pick(
            cur_X, cur_docs,
            n_neighbors=params.n_neighbors,
            n_components=params.n_components,
            n_clusters=None,
            per_cluster_topm=params.per_cluster_topm,
        )

        # 각 클러스터 요약 문서 생성
        summary_docs: List[Document] = []
        for c_idx, idxs in enumerate(tops.values()):
            summary_docs.append(_summarize_cluster(llm, cur_docs, idxs, params, level, c_idx))

        # 종료 조건: 목표 개수 이하 or 최대 레벨 도달 or 더 이상 축약 불가
        if len(summary_docs) <= params.target_k or level >= params.max_levels:
            return summary_docs

        # 다음 레벨 입력으로 요약문들을 사용 → 새로 임베딩
        cur_docs = summary_docs
        cur_X = np.asarray(emb.embed_documents([d.page_content for d in cur_docs]), dtype=np.float32)
