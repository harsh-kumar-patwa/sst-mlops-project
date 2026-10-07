"""Retrieval metrics. Deterministic and free, so they gate every pull request.

A retrieved chunk is relevant when its page (doc_path) is one of the question's source_docs.
  recall@k   share of the question's source pages found in the top k. For single-source
             questions any listed page counts (they are alternatives); multi_hop questions
             need every listed page.
  mrr@10     1 / rank of the first relevant chunk (0 if none in the top 10).
  hit@1      the very first chunk is relevant.
  section@5  a top-5 chunk is from a source page AND its heading matches source_section.
"""

from statistics import mean

from rag.evaluation.dataset import GoldenQuestion


def score_retrieval(question: GoldenQuestion, retrieved: list[dict]) -> dict:
    relevant = set(question.source_docs)
    ranks = [hit["doc_path"] for hit in retrieved]

    def recall_at(k: int) -> float:
        found = relevant & set(ranks[:k])
        if question.slice == "multi_hop":
            return len(found) / len(relevant)
        return 1.0 if found else 0.0

    first_relevant = next((i for i, doc in enumerate(ranks[:10], start=1) if doc in relevant), None)
    section = question.source_section.lower()
    section_hit = any(hit["doc_path"] in relevant and section and section in hit["section"].lower()
                      for hit in retrieved[:5])
    return {
        "recall_at_1": recall_at(1),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr_at_10": 1.0 / first_relevant if first_relevant else 0.0,
        "section_at_5": 1.0 if section_hit else 0.0,
    }


def average(rows: list[dict], keys: list[str]) -> dict:
    return {key: round(mean(row[key] for row in rows), 4) for key in keys if rows}


def by_slice(rows: list[dict], keys: list[str]) -> dict:
    slices = sorted({row["slice"] for row in rows})
    return {name: {**average([r for r in rows if r["slice"] == name], keys),
                   "n": sum(1 for r in rows if r["slice"] == name)} for name in slices}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[index], 1)
