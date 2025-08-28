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
    N = X.shape[0]
    # n_neighbors 상한/하한 안전화
    nn = n_neighbors or max(5, min(50, N - 1))
    nn = max(2, min(nn, N - 1))
    # UMAP 안전 호출 (random_state=None → n_jobs 경고 제거)
    Z = _safe_umap(X, n_neighbors=nn, n_components=n_components, random_state=None, metric="cosine")

    n_clusters = n_clusters or _estimate_n_clusters(N)
    n_clusters = max(1, min(n_clusters, N))
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hac.fit_predict(Z)

    clusters: List[List[int]] = []
    reps: List[int] = []
    tops: Dict[int, List[int]] = {}

    for c in range(n_clusters):
        idx = np.where(labels == c)[0].tolist()
        if not idx:
            continue
        centroid = X[idx].mean(axis=0)
        best_i, best_sim = None, -1.0
        for i in idx:
            sim = _cosine_sim(X[i], centroid)
            if sim > best_sim:
                best_sim, best_i = sim, i
        reps.append(int(best_i))  # type: ignore
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

# ----------------- 보조: 안전한 UMAP 적용 -----------------
def _safe_umap(X, *, n_neighbors: int = 15, n_components: int = 2, random_state=None, metric: str = "cosine"):
    X = np.asarray(X)
    n = int(X.shape[0]) if hasattr(X, "shape") else 0
    if n < 4:
        return X[:, :min(n_components, X.shape[1])] if getattr(X, "ndim", 1) == 2 else X
    nn = max(2, min(int(n_neighbors), n - 1))
    nc = max(1, min(int(n_components), n - 1))
    return umap.UMAP(
        n_neighbors=nn,
        n_components=nc,
        random_state=random_state,  # None → "n_jobs overridden" 경고 제거
        metric=metric,              # 기본 cosine (기존 동작 보존)
        verbose=False,
    ).fit_transform(X)