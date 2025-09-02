# [신규 파일] nureongi/news_chain.py
from __future__ import annotations
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from nureongi.prompt_loader import load_md_template, render_template

def build_news_chain(llm, prompt_path: str) -> Runnable:
    """
    [설명] 템플릿(.md) → LLM → 문자열 출력.
    이 체인에는 retriever/issue 템플릿이 전혀 개입하지 않음.
    """
    template = load_md_template(prompt_path)
    parser = StrOutputParser()

    def _render(inputs: dict) -> str:
        # [설명] inputs에는 title/content/url/tags/categories/meta 등 전달
        return render_template(template, **inputs)

    # [결과] 입력 dict → 렌더링된 프롬프트 → llm → 문자열
    return _render | llm | parser
