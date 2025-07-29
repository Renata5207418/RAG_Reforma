import re
import json
from pathlib import Path


def load_gold(path: Path):
    """
    Carrega um arquivo JSONL com dados de validação (gold set).
    Args:
        path (Path): Caminho para o arquivo gold.jsonl.
    Returns:
        List[Dict]: Lista de perguntas e respostas esperadas.
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def answer_matches(answer: str, must_contain: list[str]) -> bool:
    """
    Verifica se todos os termos obrigatórios aparecem na resposta.
    Args:
        answer (str): Resposta gerada pelo sistema.
        must_contain (list[str]): Lista de termos obrigatórios.
    Returns:
        bool: True se todos os termos aparecem na resposta, False caso contrário.
    """
    answer_norm = re.sub(r"\s+", " ", answer.lower())
    return all(term.lower() in answer_norm for term in must_contain)
