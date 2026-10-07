"""Builds the vector index: chunk the corpus, embed every chunk, upsert into Qdrant.

Usage: python -m rag.index [--force]
The collection name contains index_hash, so changing chunking or the embedding model
builds a new collection next to the old one instead of mixing vector versions.
"""

import argparse
import time

from qdrant_client import models

from rag.chunking import chunk_corpus
from rag.config import REPO_ROOT, collection_name, load_config
from rag.store import DENSE, SPARSE, create_collection, dense_model, get_client, point_id, sparse_model, to_sparse

BATCH_SIZE = 128        # chunks upserted to Qdrant per call
# Chunks per embedding forward pass. Attention memory grows with batch x seq_len^2: measured peak
# RSS of a full index build was >3 GB at 128 (OOM-killed on the Hugging Face build machine),
# 821 MB at 16 and 412 MB at 4. Embeddings are identical; only the pass size changes.
EMBED_BATCH_SIZE = 4


def build_index(force: bool = False) -> str:
    config = load_config()
    client = get_client()
    name = collection_name(config)

    if client.collection_exists(name):
        if not force:
            print(f"Index {name} already exists ({client.count(name).count} chunks); use --force to rebuild.")
            return name
        client.delete_collection(name)

    started = time.perf_counter()
    chunks = chunk_corpus(REPO_ROOT / config["corpus"]["path"], config["chunking"])
    embedder = dense_model(config["embedding"]["model"])
    sparse_embedder = sparse_model(config["embedding"]["sparse_model"])
    dimensions = len(next(iter(embedder.embed(["probe"]))))
    create_collection(client, name, dimensions)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [chunk.embedding_text for chunk in batch]
        dense_vectors = list(embedder.passage_embed(texts, batch_size=EMBED_BATCH_SIZE))
        sparse_vectors = list(sparse_embedder.passage_embed(texts, batch_size=EMBED_BATCH_SIZE))
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector={DENSE: dense.tolist(), SPARSE: to_sparse(sparse)},
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_path": chunk.doc_path,
                    "title": chunk.title,
                    "heading": chunk.heading,
                    "text": chunk.text,
                    "embed_model": config["embedding"]["model"],
                    "index_hash": config["index_hash"],
                    "corpus_version": config["corpus"]["version"],
                },
            )
            for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors)
        ]
        client.upsert(collection_name=name, points=points)
        print(f"  indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}", end="\r")

    print(f"\nBuilt {name}: {len(chunks)} chunks in {time.perf_counter() - started:.1f}s")
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if the index exists")
    build_index(force=parser.parse_args().force)
