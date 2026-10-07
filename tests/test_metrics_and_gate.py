from types import SimpleNamespace

from rag.evaluation.compare import compare
from rag.evaluation.dataset import GoldenQuestion
from rag.evaluation.metrics import percentile, score_retrieval
from rag.retrieve import reciprocal_rank_fusion


def question(docs, slice_name="factual", section=""):
    return GoldenQuestion("q1", "?", "", docs, section, slice_name, "test")


def retrieved(*docs):
    return [{"doc_path": doc, "section": f"Page > {doc}"} for doc in docs]


def test_alternative_sources_any_one_counts():
    scores = score_retrieval(question(["a.md", "b.md"]), retrieved("x.md", "b.md", "y.md"))
    assert scores["recall_at_1"] == 0.0
    assert scores["recall_at_5"] == 1.0
    assert scores["mrr_at_10"] == 0.5


def test_multi_hop_needs_every_source():
    scores = score_retrieval(question(["a.md", "b.md"], "multi_hop"), retrieved("a.md", "x.md"))
    assert scores["recall_at_5"] == 0.5


def test_rrf_rewards_documents_both_retrievers_agree_on():
    point = lambda pid: SimpleNamespace(id=pid)
    fused = reciprocal_rank_fusion([[point("a"), point("b")], [point("b"), point("c")]], k=60)
    assert [p.id for p, _ in fused][0] == "b"


def test_percentile_nearest_rank():
    assert percentile([100, 200, 300, 400], 50) == 200
    assert percentile([100, 200, 300, 400], 99) == 400


def results(recall, slice_recall):
    return {"meta": {}, "retrieval": {"overall": {"recall_at_5": recall, "mrr_at_10": 0.8},
                                      "by_slice": {"factual": {"recall_at_5": slice_recall}}}}


GATE = {"retrieval": {"recall_at_5": {"min": 0.75, "max_drop": 0.03},
                      "mrr_at_10": {"min": 0.5, "max_drop": 0.05}, "slice_max_drop": 0.10}}


def test_gate_passes_small_noise_and_fails_real_drop():
    assert all(c["ok"] for c in compare(results(0.90, 0.9), results(0.88, 0.85), GATE))
    failed = [c["metric"] for c in compare(results(0.90, 0.9), results(0.80, 0.6), GATE) if not c["ok"]]
    assert failed == ["recall_at_5", "recall_at_5 [factual]"]


def test_gate_floor_blocks_slow_erosion():
    checks = compare(results(0.76, 0.9), results(0.74, 0.9), GATE)
    assert not next(c for c in checks if c["metric"] == "recall_at_5")["ok"]
