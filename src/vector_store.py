from __future__ import annotations
from typing import Iterable, List
import uuid
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http.models import VectorParams, Distance
from langchain.schema import Document
from src.config import settings
import logging

log = logging.getLogger(__name__)
_client = QdrantClient(host="localhost", port=6333, timeout=30, api_key=settings.qdrant_api_key or None)


def _ensure_collection(collection: str):
    """
    Cria uma coleção no Qdrant se ela ainda não existir.
    Args:
        collection (str): Nome da coleção a ser criada.
    """
    if not _client.collection_exists(collection):
        _client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=1536,      # Tamanho padrão do embedding OpenAI (text-embedding-3-small)
                distance=Distance.COSINE,
            ),
        )


def _embedder():
    """
    Instancia o objeto de embeddings da OpenAI a partir das configs do projeto.
    Returns:
        OpenAIEmbeddings: Objeto gerador de embeddings.
    """
    return OpenAIEmbeddings(
        openai_api_key=settings.api_key,
        model=settings.embedding_model,
    )


def _get_existing_ids(collection: str):
    """
    Recupera os IDs já cadastrados na coleção Qdrant.
    Args:
        collection (str): Nome da coleção.
    Returns:
        set:Conjunto dos IDs já presentes na coleção.
    """
    hits = _client.scroll(collection_name=collection, limit=10000)[0]
    return set(hit.id for hit in hits)


def _new_store(collection: str) -> QdrantVectorStore:
    """
    Instancia um QdrantVectorStore para interação com a coleção.
    Args:
        collection (str): Nome da coleção.
    Returns:
        QdrantVectorStore: Instância conectada ao Qdrant.
    """
    return QdrantVectorStore(
        client=_client,
        collection_name=collection,
        embedding=_embedder(),
    )


# ─────────────────────────────────────────────────────────────
def initialize_vectorstore(
    docs: Iterable[Document],
    *,
    collection_name: str = "reforma_tributaria",
) -> QdrantVectorStore:
    """
    Garante que a coleção Qdrant existe e insere somente documentos novos.
    O ID de cada chunk é um UUID5 derivado do conteúdo e de um id opcional.
    Args:
        docs (Iterable[Document]): Documentos a serem indexados.
        collection_name (str): Nome da coleção a ser usada/criada.
    Returns:
        QdrantVectorStore: Instância já atualizada da coleção.
    """
    _ensure_collection(collection_name)
    store = _new_store(collection_name)

    # Descobre IDs já indexados
    existing_ids = _get_existing_ids(collection_name)

    new_docs: List[Document] = []
    for d in docs:
        raw_id = (d.metadata.get("id") or "") + d.page_content
        sha_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
        d.metadata["sha_id"] = sha_id
        if sha_id not in existing_ids:
            new_docs.append(d)

    if new_docs:
        store.add_documents(
            new_docs,
            ids=[d.metadata["sha_id"] for d in new_docs],
        )
        log.info("Qdrant: %d chunks adicionados.", len(new_docs))
    else:
        log.info("Qdrant: índice já atualizado (0 chunks novos).")

    return store
