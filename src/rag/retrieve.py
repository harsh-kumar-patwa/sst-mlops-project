"""Retrieval: dense search, optionally fused with BM25 via Reciprocal Rank Fusion.

RRF is done here in Python rather than inside Qdrant so that the dense cosine score of the
best hit stays available: that score drives the "refuse when nothing relevant" rule, and
fused RRF scores are rank-based and cannot be compared to a threshold.
"""

from dataclasses import dataclass

from rag.config import collection_name, load_config
from rag.store import DENSE, SPARSE, dense_model, get_client, sparse_model, to_sparse


class EmbeddingMismatchError(RuntimeError):
    """The index was built with a different embedding model than the one serving queries."""


@dataclass
class Hit:
    chunk_id: str
    doc_path: str
    title: str
    heading: str
    text: str
    score: float           # dense cosine similarity, or the RRF score in hybrid mode
    dense_score: float     # cosine similarity; 0.0 if the chunk was found by BM25 only

    @property
    def breadcrumb(self) -> str:
        return f"{self.title} > {self.heading}" if self.heading else self.title


class Retriever:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.client = get_client()
        self.collection = collection_name(self.config)
        self._check_index()

    def _check_index(self) -> None:
        if not self.client.collection_exists(self.collection):
            raise RuntimeError(f"Index {self.collection} not found; run `make index` first.")
        sample, _ = self.client.scroll(self.collection, limit=1, with_payload=True)
        indexed_with = sample[0].payload["embed_model"]
        serving_with = self.config["embedding"]["model"]
        if indexed_with != serving_with:
            # Mixing embedding models degrades retrieval to near-random without any error.
            raise EmbeddingMismatchError(f"index built with {indexed_with}, queries use {serving_with}")

    def _search(self, query_vector, using: str, limit: int) -> list:
        return self.client.query_points(
            self.collection, query=query_vector, using=using, limit=limit, with_payload=True
        ).points

    def search(self, question: str, top_k: int | None = None, mode: str | None = None) -> list[Hit]:
        settings = self.config["retrieval"]
        top_k = top_k or settings["top_k"]
        mode = mode or settings["mode"]
        candidates = max(settings["candidates"], top_k)

        query = next(iter(dense_model(self.config["embedding"]["model"]).query_embed(question)))
        dense_points = self._search(query.tolist(), DENSE, candidates)
        dense_scores = {point.id: point.score for point in dense_points}

        if mode == "dense":
            ranked = [(point, point.score) for point in dense_points]
        elif mode == "hybrid":
            sparse_query = next(iter(sparse_model(self.config["embedding"]["sparse_model"]).query_embed(question)))
            sparse_points = self._search(to_sparse(sparse_query), SPARSE, candidates)
            ranked = reciprocal_rank_fusion([dense_points, sparse_points], settings["rrf_k"])
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        return [
            Hit(
                chunk_id=point.payload["chunk_id"],
                doc_path=point.payload["doc_path"],
                title=point.payload["title"],
                heading=point.payload["heading"],
                text=point.payload["text"],
                score=score,
                dense_score=dense_scores.get(point.id, 0.0),
            )
            for point, score in ranked[:top_k]
        ]


def reciprocal_rank_fusion(result_lists: list[list], k: int) -> list[tuple]:
    """RRF(d) = sum over retrievers of 1 / (k + rank). Uses ranks only, so the very different
    score scales of cosine similarity and BM25 never need to be normalised against each other."""
    fused, points = {}, {}
    for results in result_lists:
        for rank, point in enumerate(results, start=1):
            fused[point.id] = fused.get(point.id, 0.0) + 1.0 / (k + rank)
            points[point.id] = point
    order = sorted(fused, key=fused.get, reverse=True)
    return [(points[point_id], fused[point_id]) for point_id in order]
