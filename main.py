from __future__ import annotations
import logging
from pathlib import Path
from src.config import settings
import time
from typing import List
from langchain.schema import Document
from src.data_loader import load_test_docs
from src.vector_store import initialize_vectorstore
from src.qa_chain import create_qa_chain
from src.utils.tone import detect_tone


# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging() -> None:
    """
        Configuração global de logging (console + arquivo) e silenciamento de libs barulhentas.
        Cria logs em ./logs/rag.log.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(settings.log_dir) / "rag.log",
                encoding="utf-8",
            ),
        ],
    )
    # ── Silencia bibliotecas de terceiros ────────────────────────
    for noisy in ("httpx", "langchain", "langchain_core", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger("langchain_core.vectorstores").setLevel(logging.ERROR)


def _docs_as_langchain(raw_docs: List[dict]) -> List[Document]:
    """
        Converte lista de dicts ({'id', 'text'}) para objetos langchain.Document com metadados.
        Args:
            raw_docs (List[dict]): Lista de dicionários com 'id' e 'text'.
        Returns:
            List[Document]: Lista de objetos Document para indexação.
    """
    return [
        Document(page_content=d["text"], metadata={"id": d["id"]})
        for d in raw_docs
    ]


def main() -> None:
    """
        Função principal: carrega dados, inicializa o pipeline e executa o loop de perguntas e respostas.
    """
    _setup_logging()
    log = logging.getLogger(__name__)

    # 1) Carrega docs de teste e transforma em Document
    raw_docs = load_test_docs()
    docs = _docs_as_langchain(raw_docs)

    # 2) Cria/atualiza o vector store
    store = initialize_vectorstore(docs)

    # 3) Prepara a chain RAG
    rag = create_qa_chain(
        store,
        k=6,  # capta mais trechos
        mmr=True,  # diversidade
        score_threshold=0.35,
    )

    # 4) Loop interativo
    print("Pipeline RAG pronto! (digite 'sair' ou Ctrl+C para finalizar)")
    try:
        while True:
            pergunta = input("Pergunta> ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in ("sair", "exit", "quit"):
                break

            tone = detect_tone(pergunta)
            t0 = time.perf_counter()
            result = rag.invoke({"query": pergunta, "tone": tone})
            dt = time.perf_counter() - t0

            resposta = result["result"]
            print("\nResposta:", resposta)
            print(f"( {dt:.2f}s – tom detectado: {tone})")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\nEncerrando… até a próxima!")
    except Exception as exc:
        log.exception("Erro inesperado no loop principal: %s", exc)
        print("Ocorreu um erro inesperado. Veja o log para detalhes.")


if __name__ == "__main__":
    main()
