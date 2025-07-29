import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import itertools
from pathlib import Path
from langchain.schema import Document
from src.data_loader import load_test_docs
from src.vector_store import initialize_vectorstore
from src.qa_chain import create_qa_chain
from tests.utils import load_gold, answer_matches


def docs_as_langchain(raw):
    """
       Converte lista de dicts para langchain.Document.
    """
    return [Document(page_content=d["text"], metadata={"id": d["id"]}) for d in raw]


def precision_for_params(k, thr):
    """
    Avalia a precisão para uma combinação de parâmetros k e threshold.
    Args:
        k (int): Número de trechos recuperados.
        thr (float): Limite mínimo de similaridade.
    Returns:
        float: Precisão obtida no gold set.
    """
    docs = docs_as_langchain(load_test_docs())
    store = initialize_vectorstore(
        docs,  # 1º posicional
        persist_dir=".calib_vec",  # ← keyword
        collection_name="calib",  # ← keyword
    )

    rag = create_qa_chain(store, k=k, mmr=True, score_threshold=thr)
    gold = load_gold(Path(__file__).parent / "data" / "gold.jsonl")
    hits = 0
    for item in gold:
        out = rag.invoke({"query": item["question"], "tone": "objetivo"})
        if answer_matches(out["result"], item["ideal_answer_contains"]):
            hits += 1
    return hits / len(gold)


grid_k = [4, 6, 8]
grid_thr = [0.25, 0.3, 0.35, 0.4]
best = (0, None, None)

for k, thr in itertools.product(grid_k, grid_thr):
    prec = precision_for_params(k, thr)
    print(f"k={k}, thr={thr:.2f} -> {prec:.2%}")
    if prec > best[0]:
        best = (prec, k, thr)

print("\nMelhor combinação:", best)
