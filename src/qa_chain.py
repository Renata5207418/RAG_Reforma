from __future__ import annotations
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStore
from langchain_openai import ChatOpenAI
from src.config import settings
from src.qa_safe import SafeRetrievalQA


# ──────────────────── Prompt ─────────────────────────────────────────────────
TEMPLATE = """\
Você é um assistente tributário. Utilize **somente** as informações fornecidas no contexto
para responder à pergunta do usuário. Se a informação **não** estiver no contexto,
responda exatamente: **"Desculpe, não sei essa informação."**

Adote o tom {tone}.

=== CONTEXTO
{context}
=== FIM DO CONTEXTO

Pergunta do usuário:
{question}

Sua resposta (em português):"""
PROMPT = PromptTemplate(
    input_variables=["context", "question", "tone"],
    template=TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
def create_qa_chain(
    vectorstore: VectorStore,
    *,
    model_name: str | None = None,
    k: int = 4,
    score_threshold: float = 0.35,
    chain_type: str = "stuff",
    mmr: bool = False,
    stream: bool = False,
) -> SafeRetrievalQA:
    """
    Retorna uma RetrievalQA já configurada.

    Args:
        vectorstore: Base de vetores (Chroma ou equivalente).
        model_name : Override do modelo; default vem de settings.
        k          : Nº de trechos a recuperar.
        score_threshold: Similaridade mínima (0‑1) para aceitar chunk.
        chain_type : 'stuff'|'map_reduce'|'refine' (ver docs LangChain).
        mmr        : Se True, usa busca Max‑Marginal‑Relevance.
        stream     : Se True, ativa streaming de tokens no ChatOpenAI.

    Raises:
        ValueError se não houver docs relevantes (condição verificada
        dentro do retriever wrapper).
    """
    # 1) Retriever
    search_kwargs = {"k": k, "score_threshold": score_threshold}
    retriever = vectorstore.as_retriever(
        search_type="mmr" if mmr else "similarity_score_threshold",
        search_kwargs=search_kwargs,
    )

    # 2) LLM
    llm = ChatOpenAI(
        model=model_name or settings.default_model,
        openai_api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        top_p=settings.top_p,
        streaming=stream,
    )

    # 3) QA Chain (retorna docs também)
    qa = SafeRetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa
