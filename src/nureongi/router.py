from dotenv import load_dotenv ; load_dotenv()
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from nureongi.persona import PersonaSlug, PROMPT_REGISTRY, ALIAS_TO_SLUG

router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ROUTER_LABELS = ", ".join(s.value for s in PROMPT_REGISTRY.keys())

router_prompt = PromptTemplate.from_template(
    """다음 질문을 가장 적합한 라벨(슬러그) 하나로 분류하세요.
후보: {labels}
규칙:
- 후보 중 하나만 출력(여분 텍스트 금지).
질문: {question}
"""
)

def _canonical_slug(s: Optional[str]) -> Optional[PersonaSlug]:
    if not s:
        return None
    return ALIAS_TO_SLUG.get(s.strip().lower())

def route_persona(question: str) -> PersonaSlug:
    msg = router_prompt.format(labels=ROUTER_LABELS, question=question)
    out = router_llm.invoke(msg).content.strip()
    slug = _canonical_slug(out)
    return slug or PersonaSlug.STAFF_REPORT

def choose_prompt(question: str, context: str, persona: str | None = "auto") -> str:
    if persona == "auto":
        slug = route_persona(question)
    else:
        slug = _canonical_slug(persona) or PersonaSlug.STAFF_REPORT
    tmpl = PROMPT_REGISTRY[slug].template
    return tmpl.format(question=question, context=context)