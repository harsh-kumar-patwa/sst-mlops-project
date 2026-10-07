"""Runs the evaluation suite against the golden set and writes a results JSON.

Usage:
  python -m rag.evaluation.run                         # retrieval + generation (needs API keys)
  python -m rag.evaluation.run --retrieval-only        # free, deterministic, no keys needed
  python -m rag.evaluation.run --golden eval/smoke.csv --out eval/results/smoke.json
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from rag.config import REPO_ROOT, load_config
from rag.evaluation.dataset import load_golden
from rag.evaluation.metrics import average, by_slice, percentile, score_retrieval
from rag.retrieve import Retriever

RETRIEVAL_KEYS = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "section_at_5"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def evaluate_retrieval(questions, config) -> tuple[dict, list[dict]]:
    retriever = Retriever(config)
    rows = []
    for question in (q for q in questions if q.answerable):
        hits = retriever.search(question.question, top_k=10)
        retrieved = [{"doc_path": h.doc_path, "section": h.breadcrumb, "score": round(h.score, 4)} for h in hits]
        rows.append({"id": question.id, "slice": question.slice,
                     **score_retrieval(question, retrieved), "top_docs": [r["doc_path"] for r in retrieved[:5]]})
    return {"overall": average(rows, RETRIEVAL_KEYS), "by_slice": by_slice(rows, RETRIEVAL_KEYS), "n": len(rows)}, rows


def evaluate_generation(questions, config) -> tuple[dict, list[dict]]:
    from rag.evaluation.judge import Judge
    from rag.pipeline import RagPipeline

    pipeline, judge = RagPipeline(config), Judge(config)
    rows = []
    for number, question in enumerate(questions, start=1):
        print(f"  generating {number}/{len(questions)}: {question.id}", end="\r")
        result = pipeline.answer(question.question, use_cache=True, include_context=True)
        refused = result.status == "refused"
        row = {
            "id": question.id, "slice": question.slice, "status": result.status, "answer": result.answer,
            "refused": refused, "citations": len(result.citations), "invalid_citations": result.invalid_citations,
            "cost_usd": result.cost_usd, "latency_ms": result.latency_ms["total"],
            "cached": result.extras.get("cached", False),
        }
        if result.status == "answered":
            row["faithfulness"], row["claims"] = judge.faithfulness(result.answer, result.extras["context"])
            row["citation_valid"] = 1.0 if result.citations and not result.invalid_citations else 0.0
        if question.answerable:
            row["correctness"], row["correctness_reason"] = (
                judge.correctness(question.question, question.reference_answer, result.answer)
                if result.status == "answered" else (0.0, f"not answered ({result.status})"))
        else:
            row["refusal_correct"] = 1.0 if refused else 0.0
        rows.append(row)
    print()

    answered = [r for r in rows if r["status"] == "answered"]
    answerable = [r for r in rows if "correctness" in r]
    unanswerable = [r for r in rows if "refusal_correct" in r]
    fresh_latencies = [r["latency_ms"] for r in rows if not r["cached"] and r["status"] != "refused"]

    def mean_of(subset, key):
        return round(mean(r[key] for r in subset), 4) if subset else None

    overall = {
        "faithfulness": mean_of(answered, "faithfulness"),
        "correctness": mean_of(answerable, "correctness"),
        "refusal_accuracy": mean_of(unanswerable, "refusal_correct"),
        "false_refusal_rate": round(sum(r["refused"] for r in answerable) / len(answerable), 4) if answerable else None,
        "citation_validity": mean_of(answered, "citation_valid"),
        "degraded": sum(1 for r in rows if r["status"] == "degraded"),
    }
    slices = {}
    for name in sorted({r["slice"] for r in rows}):
        subset = [r for r in rows if r["slice"] == name]
        slices[name] = {"n": len(subset),
                        "faithfulness": mean_of([r for r in subset if "faithfulness" in r], "faithfulness"),
                        "correctness": mean_of([r for r in subset if "correctness" in r], "correctness")}
    summary = {
        "overall": overall, "by_slice": slices, "n": len(rows),
        "cost": {"generation_per_query_usd": round(mean(r["cost_usd"] for r in rows), 6),
                 "judge_spent_this_run_usd": round(judge.cost_usd, 4)},
        "latency_ms": {"p50": percentile(fresh_latencies, 50), "p95": percentile(fresh_latencies, 95),
                       "p99": percentile(fresh_latencies, 99), "n_uncached": len(fresh_latencies)},
    }
    return summary, rows


def run_eval(golden: Path, config: dict | None = None, retrieval_only: bool = False) -> dict:
    config = config or load_config()
    questions = load_golden(golden)
    started = time.perf_counter()
    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git_sha(), "n_questions": len(questions),
            # Repo-relative, so local runs and CI label the same question set identically.
            "golden": golden.resolve().relative_to(REPO_ROOT).as_posix(),
            "config_hash": config["config_hash"], "index_hash": config["index_hash"],
            "retrieval_mode": config["retrieval"]["mode"], "chunk_size": config["chunking"]["chunk_size_tokens"],
            "prompt_version": config["generation"]["prompt_version"], "generator": config["generation"]["model"],
            "judge": config["judge"]["model"],
        },
    }
    results["retrieval"], retrieval_rows = evaluate_retrieval(questions, config)
    results["per_question"] = {"retrieval": retrieval_rows}
    if not retrieval_only:
        results["generation"], generation_rows = evaluate_generation(questions, config)
        results["per_question"]["generation"] = generation_rows
    results["meta"]["duration_s"] = round(time.perf_counter() - started, 1)
    return results


def print_summary(results: dict) -> None:
    meta = results["meta"]
    print(f"\nconfig {meta['config_hash']} | {meta['retrieval_mode']} | chunk {meta['chunk_size']} | "
          f"{meta['prompt_version']} | {meta['n_questions']} questions | {meta['duration_s']}s")
    print("retrieval:", "  ".join(f"{k}={v:.3f}" for k, v in results["retrieval"]["overall"].items()))
    for name, values in results["retrieval"]["by_slice"].items():
        print(f"  {name:<13} n={values['n']:<3} recall@5={values['recall_at_5']:.3f}  mrr@10={values['mrr_at_10']:.3f}")
    if "generation" in results:
        generation = results["generation"]
        print("generation:", "  ".join(f"{k}={v}" for k, v in generation["overall"].items()))
        print("cost:", generation["cost"], "latency:", generation["latency_ms"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=REPO_ROOT / "eval" / "golden.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results" / "latest.json")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    results = run_eval(args.golden, retrieval_only=args.retrieval_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print_summary(results)
    print(f"\nwrote {args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out}")


if __name__ == "__main__":
    main()
