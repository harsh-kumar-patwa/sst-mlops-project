"""Embedding models and the Qdrant vector store.

Local development and CI use Qdrant's embedded mode (an on-disk folder, no server, no key).
The deployed service sets QDRANT_URL / QDRANT_API_KEY to use Qdrant Cloud with the same code.
Every collection stores both a dense vector (semantic match) and a BM25 sparse vector
(exact-token match), so switching retrieval.mode between dense and hybrid needs no re-index.
"""

import atexit
import os
import uuid
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from rag.config import REPO_ROOT

LOCAL_PATH = REPO_ROOT / ".qdrant"
DENSE, SPARSE = "dense", "bm25"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if url:
        client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=10)
    else:
        client = QdrantClient(path=str(LOCAL_PATH))
    # Embedded mode holds a file lock; release it before interpreter teardown.
    atexit.register(client.close)
    return client


@lru_cache(maxsize=2)
def dense_model(name: str) -> TextEmbedding:
    return TextEmbedding(model_name=name)


@lru_cache(maxsize=2)
def sparse_model(name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=name)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def create_collection(client: QdrantClient, name: str, dimensions: int) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dimensions, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def to_sparse(embedding) -> models.SparseVector:
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
