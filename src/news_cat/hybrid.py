# src/news_cat/hybrid.py
from __future__ import annotations
import os, re, json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import google.generativeai as genai

CATEGORIES = [
    {"slug": "research", "label_kr": "Research (학술)"},
    {"slug": "technology_product", "label_kr": "Technology & Product (기술/제품)"},
    {"slug": "market_corporate", "label_kr": "Market & Corporate (시장/기업)"},
    {"slug": "policy_regulation", "label_kr": "Policy & Regulation (정책/규제)"},
    {"slug": "society_culture", "label_kr": "Society & Culture (사회/문화)"},
    {"slug": "incidents_safety", "label_kr": "Incidents & Safety (사건/안전/운영)"},
]
CAT_SLUGS = [c["slug"] for c in CATEGORIES]
LABEL_KR = {c["slug"]: c["label_kr"] for c in CATEGORIES}

PROTOS = {
    "research": [
        "논문, 프리프린트, 학회 채택과 수상, 벤치마크나 데이터셋 공개 소식",
        "학술 연구 성과가 핵심인 뉴스",
    ],
    "technology_product": [
        "모델이나 제품 릴리스, 기능 업데이트, 모델 카드 변경, 엔지니어링 기능 공지",
        "제품과 기술 업데이트 중심의 뉴스",
    ],
    "market_corporate": [
        "투자, M&A, IPO, 실적 발표, 리더십이나 조직 개편, 전략 제휴나 상용 계약",
        "기업의 자금 거래와 상업 전략 관련 소식",
    ],
    "policy_regulation": [
        "법과 규제, 가이드라인, 공공자금 지원, 수출통제, 표준화와 거버넌스",
        "정부나 공공기관의 정책 변화 관련 뉴스",
    ],
    "society_culture": [
        "대중 활용 트렌드, 창작과 교육, 밈, 저작권이나 윤리 공론",
        "사회적 파급과 수용이 중심인 뉴스",
    ],
    "incidents_safety": [
        "서비스 장애, 보안 사고와 데이터 유출, 오남용, 서비스 중단이나 리콜",
        "운영과 안전 사고 및 대응 관련 소식",
    ],
}

SLUG_RE = re.compile(r"(research|technology_product|market_corporate|policy_regulation|society_culture|incidents_safety)")

def _fallback_rule(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["arxiv","neurips","icml","iclr","preprint","dataset"]): return "research"
    if any(k in t for k in ["outage","incident","security","breach","leak","cve","downtime","attack"]): return "incidents_safety"
    if any(k in t for k in ["act","law","regulation","policy","guideline","rfp"]): return "policy_regulation"
    if any(k in t for k in ["funding","m&a","acquisition","ipo","earnings","revenue","partnership"]): return "market_corporate"
    if any(k in t for k in ["trend","meme","education","teacher","artists","culture","ethics","copyright"]): return "society_culture"
    return "technology_product"

class HybridClassifier:
    def __init__(
        self,
        emb_model_name: Optional[str] = None,
        gemini_model_name: Optional[str] = None,
        hf_home: Optional[str] = None,
    ):
        # 캐시 루트 강제(원하면 생략 가능)
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
            os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
            os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

        self.emb_model_name = emb_model_name or os.getenv("EMB_MODEL", "BAAI/bge-base-en-v1.5")
        self.emb = HuggingFaceBgeEmbeddings(
            model_name=self.emb_model_name,
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages:",
            embed_instruction="Represent this document for retrieval:",
        )
        self.proto = self._build_prototypes()

        # Gemini
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_model_name = gemini_model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini = genai.GenerativeModel(self.gemini_model_name)
        else:
            self.gemini = None

    @classmethod
    def from_env(cls):
        return cls(
            emb_model_name=os.getenv("EMB_MODEL"),
            gemini_model_name=os.getenv("GEMINI_MODEL"),
            hf_home=os.getenv("HF_HOME"),
        )

    def _build_prototypes(self) -> np.ndarray:
        texts, counts = [], []
        for slug in CAT_SLUGS:
            lst = PROTOS[slug]
            texts.extend(lst); counts.append(len(lst))
        mat = np.array(self.emb.embed_documents(texts), dtype=np.float32)  # 정규화됨
        vecs, s = [], 0
        for n in counts:
            v = mat[s:s+n].mean(axis=0); s += n
            v = v / (np.linalg.norm(v) + 1e-12)
            vecs.append(v)
        return np.stack(vecs, axis=0)  # (6,d)

    def _embed_pick(self, text: str) -> Tuple[str, float]:
        if not text or len(text.strip()) < 10:
            return _fallback_rule(text), 0.0
        q = np.array(self.emb.embed_query(text.strip()), dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-12)
        sims = q @ self.proto.T
        order = sims.argsort()[::-1]
        top, second = float(sims[order[0]]), float(sims[order[1]])
        slug = CAT_SLUGS[int(order[0])]
        conf_top = (top + 1.0) / 2.0
        margin = max(0.0, top - second)
        conf_margin = min(1.0, margin / 0.4)
        conf = round(0.7 * conf_top + 0.3 * conf_margin, 2)
        return slug, conf

    def _llm_pick(self, text: str, meta: Dict[str, Any]) -> Tuple[str, float, str]:
        if self.gemini is None:
            return "", 0.0, "llm_unavailable"
        title = meta.get("title",""); source = meta.get("source","")
        date = meta.get("pub_date") or meta.get("published_at","")
        link = meta.get("link") or meta.get("guid","")
        prompt = (
            "You are a strict news category classifier. Choose exactly ONE slug from this set:\n"
            "research | technology_product | market_corporate | policy_regulation | society_culture | incidents_safety\n\n"
            "Return ONLY the slug text, no quotes, no explanation.\n\n"
            f"TITLE: {title}\nSOURCE: {source}\nDATE: {date}\nURL: {link}\n\n"
            f"CONTENT:\n{(text or '')[:2000]}\n"
        )
        try:
            resp = self.gemini.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 8})
            out = (resp.text or "").strip().lower()
            m = SLUG_RE.search(out)
            if not m:
                return "", 0.0, "llm_parse_fail"
            return m.group(1), 0.65, "llm(flash)"
        except Exception as e:
            return "", 0.0, f"llm_error:{type(e).__name__}"

    def classify_rows(
        self,
        rows: List[Dict[str, Any]],
        max_chars: int = 2000,
        margin_thr: float = 0.15,
        top1_thr: float = 0.30,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            text = (r.get("content") or r.get("description") or r.get("title") or "").strip()
            if len(text) > max_chars:
                text = text[:max_chars]
            slug, emb_conf = self._embed_pick(text)

            rationale = "embedding(centroid)"
            conf = emb_conf
            # 저신뢰면 LLM 보정
            # emb_conf는 top/마진을 합쳐놓은 값이라, 필요하면 별도 top,margin을 저장해도 됨
            if emb_conf < 0.45 and self.gemini is not None:
                g_slug, g_conf, g_rat = self._llm_pick(text, r)
                if g_slug in LABEL_KR:
                    slug, conf, rationale = g_slug, max(emb_conf, g_conf), g_rat

            enriched = dict(r)
            enriched["category_pred"] = slug
            enriched["category_label_kr"] = LABEL_KR[slug]
            enriched["category_confidence"] = round(float(conf), 2)
            enriched["category_rationale"] = rationale
            out.append(enriched)
        return out
