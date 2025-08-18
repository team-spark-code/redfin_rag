# RAG 체인(라우터 + 포맷터 결합)
from dotenv import load_dotenv ; load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from nureongi.format import format_ctx
from nureongi.router import choose_prompt

def build_routed_rag_chain(retriever, model: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = ChatOpenAI(model=model, temperature=temperature)

    # 입력 정규화: 문자열 또는 딕셔너리 모두 처리
    def normalize_input(x):
        if isinstance(x, str):
            return {"question": x, "persona": "auto"}
        return x

    # 컨텍스트 추가
    def with_context(x: dict):
        ctx = format_ctx(retriever.get_relevant_documents(x["question"]))
        return {"question": x["question"], "persona": x.get("persona", "auto"), "context": ctx}

    # 최종 프롬프트 생성
    def to_final_prompt(x: dict) -> str:
        return choose_prompt(x["question"], x["context"], x["persona"])

    chain = (
        RunnableLambda(normalize_input)
        | RunnableLambda(with_context)
        | RunnableLambda(to_final_prompt)
        | llm
        | StrOutputParser()
    )
    return chain
