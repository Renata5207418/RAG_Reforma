import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import pytest
import time
from pathlib import Path
from src.data_loader import load_test_docs
from src.vector_store import initialize_vectorstore
from src.qa_chain import create_qa_chain
from tests.utils import load_gold, answer_matches
from langchain.schema import Document


def docs_as_langchain(raw_docs):
    """
    Converte lista de dicts para langchain.Document.
    """
    return [
        Document(page_content=d["text"], metadata={"id": d["id"]})
        for d in raw_docs
    ]


@pytest.fixture(scope="session")
def rag():
    """
    Inicializa o pipeline RAG com dados de teste e armazena o vetor em local temporário.
    """
    raw = load_test_docs()  # lista de dicts
    docs = docs_as_langchain(raw)

    store = initialize_vectorstore(
        docs=docs,
        persist_dir=".pytest_vector",  # separa do índice oficial
        collection_name="pytest_rt",
    )
    return create_qa_chain(store, k=6, mmr=True, score_threshold=0.35)


def test_precision_at_1(rag):
    """
    Testa se o pipeline atinge pelo menos 80% de precisão no gold set.
    """
    gold = load_gold(Path(__file__).parent / "data" / "gold.jsonl")
    hits = 0
    latencies = []

    for item in gold:
        t0 = time.perf_counter()
        out = rag.invoke({"query": item["question"], "tone": "objetivo"})
        latencies.append(time.perf_counter() - t0)
        if answer_matches(out["result"], item["ideal_answer_contains"]):
            hits += 1

    precision = hits / len(gold)
    avg_latency = sum(latencies) / len(latencies)
    print(f"Precision@1: {precision:.2%} | Latência média: {avg_latency:.2f}s")

    assert precision >= 0.8, "Precision caiu abaixo do limiar mínimo (80%)"
