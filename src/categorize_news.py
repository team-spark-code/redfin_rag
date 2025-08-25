#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

from news_cat.hybrid import HybridClassifier

def load_jsonl(path: Path, max_rows: Optional[int]=None) -> List[Dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except: continue
            if max_rows and len(rows) >= max_rows: break
    return rows

def write_jsonl(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/raw_contents.json")
    ap.add_argument("--output", default="output/categorized_hybrid.jsonl")
    ap.add_argument("--max-rows", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--emb-conf-thr", type=float, default=0.45)  # 필요시 사용
    args = ap.parse_args()

    inp, outp = Path(args.input), Path(args.output)
    if not inp.exists():
        print(f"[ERROR] 입력 없음: {inp}", file=sys.stderr); sys.exit(1)

    rows = load_jsonl(inp, max_rows=args.max_rows)
    clf = HybridClassifier.from_env()
    results = clf.classify_rows(rows, max_chars=args.max_chars)
    write_jsonl(results, outp)
    print(f"[OK] {outp} 저장 ({len(results)}건)")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# 하이브리드 카테고리 분류기
# - 1차: BGE 임베딩(centroid 최근접)
# - 2차: 저신뢰 문서만 Gemini Flash로 재판정
# 입력:  src/dataset/raw_contents.jsonl
# 출력:  src/output/categorized_hybrid.jsonl
# 처리:  기본 20건 (--max-rows로 조절)
# """

# from __future__ import annotations
# import os, sys, json, argparse, re
# from pathlib import Path
# from typing import Dict, Any, List, Optional

# # .env 로딩
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override=False)

# import numpy as np
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# # (옵션) Gemini 사용
# import google.generativeai as genai

# # ================== 캐시 경로 (D: 기본) ==================
# DEFAULT_CACHE_ROOT = os.getenv("HF_HOME") or r"D:\hf_cache"
# Path(DEFAULT_CACHE_ROOT).mkdir(parents=True, exist_ok=True)
# os.environ.setdefault("HF_HOME", DEFAULT_CACHE_ROOT)
# os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(DEFAULT_CACHE_ROOT) / "transformers"))
# os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(DEFAULT_CACHE_ROOT) / "hub"))
# os.environ.setdefault("HF_DATASETS_CACHE", str(Path(DEFAULT_CACHE_ROOT) / "datasets"))

# # ================== 카테고리/프로토타입 ==================
# CATEGORIES = [
#     {"slug": "research",            "label_kr": "Research (학술)"},
#     {"slug": "technology_product",  "label_kr": "Technology & Product (기술/제품)"},
#     {"slug": "market_corporate",    "label_kr": "Market & Corporate (시장/기업)"},
#     {"slug": "policy_regulation",   "label_kr": "Policy & Regulation (정책/규제)"},
#     {"slug": "society_culture",     "label_kr": "Society & Culture (사회/문화)"},
#     {"slug": "incidents_safety",    "label_kr": "Incidents & Safety (사건/안전/운영)"},
# ]
# CAT_SLUGS = [c["slug"] for c in CATEGORIES]
# SLUG_SET = set(CAT_SLUGS)

# PROTOS = {
#     "research": [
#         "논문, 프리프린트, 학회 채택과 수상, 벤치마크나 데이터셋 공개 소식",
#         "학술 연구 성과가 핵심인 뉴스"
#     ],
#     "technology_product": [
#         "모델이나 제품 릴리스, 기능 업데이트, 모델 카드 변경, 엔지니어링 기능 공지",
#         "제품과 기술 업데이트 중심의 뉴스"
#     ],
#     "market_corporate": [
#         "투자, M&A, IPO, 실적 발표, 리더십이나 조직 개편, 전략 제휴나 상용 계약",
#         "기업의 자금 거래와 상업 전략 관련 소식"
#     ],
#     "policy_regulation": [
#         "법과 규제, 가이드라인, 공공자금 지원, 수출통제, 표준화와 거버넌스",
#         "정부나 공공기관의 정책 변화 관련 뉴스"
#     ],
#     "society_culture": [
#         "대중 활용 트렌드, 창작과 교육, 밈, 저작권이나 윤리 공론",
#         "사회적 파급과 수용이 중심인 뉴스"
#     ],
#     "incidents_safety": [
#         "서비스 장애, 보안 사고와 데이터 유출, 오남용, 서비스 중단이나 리콜",
#         "운영과 안전 사고 및 대응 관련 소식"
#     ],
# }

# # ================== 유틸 ==================
# def load_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
#     rows = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rows.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue
#             if max_rows and len(rows) >= max_rows:
#                 break
#     return rows

# def ensure_output_dir(path: Path):
#     path.parent.mkdir(parents=True, exist_ok=True)

# def write_jsonl(rows: List[Dict[str, Any]], out_path: Path):
#     ensure_output_dir(out_path)
#     with out_path.open("w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

# def rule_fallback(item: Dict[str, Any]) -> str:
#     text = " ".join([
#         item.get("title") or "",
#         item.get("description") or "",
#         item.get("source") or "",
#         item.get("link") or "",
#     ]).lower()
#     if any(k in text for k in ["arxiv", "aclweb", "neurips", "icml", "iclr", "paper", "preprint", "dataset"]):
#         return "research"
#     if any(k in text for k in ["outage", "incident", "security", "breach", "leak", "cve", "downtime", "attack"]):
#         return "incidents_safety"
#     if any(k in text for k in ["act", "law", "regulation", "policy", "compliance", "ai act", "guideline", "rfp"]):
#         return "policy_regulation"
#     if any(k in text for k in ["funding", "raised", "m&a", "acquisition", "merger", "ipo", "earnings", "revenue", "partnership"]):
#         return "market_corporate"
#     if any(k in text for k in ["trend", "meme", "education", "teacher", "artists", "culture", "ethics", "copyright"]):
#         return "society_culture"
#     return "technology_product"

# # ================== 임베딩/프로토타입 ==================
# def build_bge() -> HuggingFaceBgeEmbeddings:
#     model_name = os.getenv("EMB_MODEL", "BAAI/bge-base-en-v1.5")
#     return HuggingFaceBgeEmbeddings(
#         model_name=model_name,
#         encode_kwargs={"normalize_embeddings": True},
#         query_instruction="Represent this sentence for searching relevant passages:",
#         embed_instruction="Represent this document for retrieval:",
#     )

# def build_category_prototypes(emb: HuggingFaceBgeEmbeddings) -> np.ndarray:
#     all_texts, counts = [], []
#     for slug in CAT_SLUGS:
#         lst = PROTOS[slug]
#         all_texts.extend(lst); counts.append(len(lst))
#     mat = np.array(emb.embed_documents(all_texts), dtype=np.float32)  # (sum_n, d) 정규화됨
#     vecs, s = [], 0
#     for n in counts:
#         v = mat[s:s+n].mean(axis=0)
#         v = v / (np.linalg.norm(v) + 1e-12)
#         vecs.append(v)
#         s += n
#     return np.stack(vecs, axis=0)  # (6, d)

# # ================== Gemini (LLM 폴백) ==================
# def build_gemini():
#     api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#     model_id = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
#     if not api_key:
#         return None, None
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel(model_id)
#     return model, model_id

# SLUG_RE = re.compile(r"(research|technology_product|market_corporate|policy_regulation|society_culture|incidents_safety)")

# def gemini_pick_category(model, text: str, meta: Dict[str, str]) -> tuple[str, float, str]:
#     """
#     Gemini에 '슬러그 하나만 출력' 요청 → 파싱 → (slug, conf, rationale)
#     """
#     if model is None:
#         return "", 0.0, "llm_unavailable"

#     title = meta.get("title","")
#     source = meta.get("source","")
#     date = meta.get("pub_date","") or meta.get("published_at","")
#     link = meta.get("link","") or meta.get("guid","")

#     prompt = (
#         "You are a strict news category classifier. Choose exactly ONE slug from this set:\n"
#         "research | technology_product | market_corporate | policy_regulation | society_culture | incidents_safety\n\n"
#         "Return ONLY the slug text, no quotes, no explanation.\n\n"
#         f"TITLE: {title}\nSOURCE: {source}\nDATE: {date}\nURL: {link}\n\n"
#         f"CONTENT:\n{text[:2000]}\n"
#     )
#     try:
#         resp = model.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 8})
#         out = (resp.text or "").strip().lower()
#         m = SLUG_RE.search(out)
#         if not m:
#             return "", 0.0, "llm_parse_fail"
#         slug = m.group(1)
#         return slug, 0.65, "llm(flash)"
#     except Exception as e:
#         return "", 0.0, f"llm_error:{type(e).__name__}"

# # ================== 하이브리드 분류 ==================
# def classify_hybrid(
#     rows: List[Dict[str, Any]],
#     max_chars: int = 2000,
#     margin_thr: float = 0.15,      # top1 - top2 < 0.15 → 저신뢰
#     top1_thr: float = 0.30         # top1 코사인 < 0.30 → 저신뢰
# ) -> List[Dict[str, Any]]:
#     emb = build_bge()
#     proto = build_category_prototypes(emb)
#     gemini_model, gemini_name = build_gemini()

#     # 1) 텍스트 준비
#     texts, metas = [], []
#     for r in rows:
#         t = (r.get("content") or r.get("description") or r.get("title") or "").strip()
#         texts.append(t[:max_chars] if len(t) > max_chars else t)
#         metas.append({
#             "title": r.get("title",""), "source": r.get("source",""),
#             "pub_date": r.get("pub_date") or r.get("published_at",""),
#             "link": r.get("link") or r.get("guid","")
#         })

#     # 2) 비어있지 않은 것만 배치 임베딩
#     empty_mask = [len(t) < 10 for t in texts]
#     idxs = [i for i, e in enumerate(empty_mask) if not e]
#     if idxs:
#         doc_vecs = np.array(emb.embed_documents([texts[i] for i in idxs]), dtype=np.float32)  # (m,d)
#         sims = doc_vecs @ proto.T  # (m,6)
#     else:
#         sims = np.zeros((0, len(CAT_SLUGS)), dtype=np.float32)

#     # 3) 병합 + 하이브리드 라우팅
#     out_rows: List[Dict[str, Any]] = []
#     ptr = 0
#     for i, row in enumerate(rows):
#         if empty_mask[i]:
#             slug = rule_fallback(row)
#             conf = 0.0
#             rationale = "fallback(rule)"
#         else:
#             srow = sims[ptr]
#             order = srow.argsort()[::-1]
#             top, second = float(srow[order[0]]), float(srow[order[1]])
#             slug = CAT_SLUGS[int(order[0])]
#             # 임베딩 신뢰도
#             conf_top = (top + 1.0) / 2.0
#             margin = max(0.0, top - second)
#             conf_margin = min(1.0, margin / 0.4)
#             emb_conf = 0.7 * conf_top + 0.3 * conf_margin

#             # 저신뢰이면 LLM 보정 시도
#             if (margin < margin_thr or top < top1_thr) and gemini_model is not None:
#                 g_slug, g_conf, g_rat = gemini_pick_category(gemini_model, texts[i], metas[i])
#                 if g_slug in SLUG_SET:
#                     slug, conf, rationale = g_slug, max(emb_conf, g_conf), g_rat
#                 else:
#                     conf, rationale = round(emb_conf, 2), "embedding(centroid)"  # LLM 실패 → 임베딩 유지
#             else:
#                 conf, rationale = round(emb_conf, 2), "embedding(centroid)"
#             ptr += 1

#         enriched = dict(row)
#         enriched["category_pred"] = slug
#         enriched["category_label_kr"] = next(c["label_kr"] for c in CATEGORIES if c["slug"] == slug)
#         enriched["category_confidence"] = round(float(conf), 2)
#         enriched["category_rationale"] = rationale
#         out_rows.append(enriched)

#     return out_rows

# # ================== CLI ==================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input", type=str, default="dataset/raw_contents.json")
#     ap.add_argument("--output", type=str, default="output/categorized_hybrid.jsonl")
#     ap.add_argument("--max-rows", type=int, default=20, help="처리할 최대 기사 수(기본 20)")
#     ap.add_argument("--max-chars", type=int, default=2000, help="본문 최대 사용 길이(기본 2000자)")
#     ap.add_argument("--margin-thr", type=float, default=0.15)
#     ap.add_argument("--top1-thr", type=float, default=0.30)
#     args = ap.parse_args()

#     in_path = Path(args.input)
#     out_path = Path(args.output)
#     if not in_path.exists():
#         print(f"[ERROR] 입력 파일이 없습니다: {in_path}", file=sys.stderr)
#         sys.exit(1)

#     rows = load_jsonl(in_path, max_rows=(args.max_rows or None))
#     print(f"[INFO] 읽은 문서 수: {len(rows)} from {in_path}")
#     print(f"[INFO] EMB_MODEL: {os.getenv('EMB_MODEL', 'BAAI/bge-base-en-v1.5')}")
#     print(f"[INFO] GEMINI_MODEL: {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')} (키 {'있음' if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') else '없음'})")
#     print(f"[INFO] HF cache: {DEFAULT_CACHE_ROOT}")

#     results = classify_hybrid(
#         rows,
#         max_chars=args.max_chars,
#         margin_thr=args.margin_thr,
#         top1_thr=args.top1_thr,
#     )
#     write_jsonl(results, out_path)
#     print(f"[OK] 저장 완료: {out_path} (총 {len(results)}건)")

# if __name__ == "__main__":
#     main()
