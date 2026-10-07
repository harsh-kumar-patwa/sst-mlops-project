"""Request logging for monitoring. Runs as a FastAPI background task, after the response is sent,
so observability adds no latency to the user's request.

logs/requests.jsonl   one row per /ask call: latency per stage, tokens, cost, status, versions
logs/feedback.jsonl   thumbs up/down from users (the online north-star signal)
logs/online_scores.jsonl  sampled LLM-judge faithfulness on live answers (online quality)
"""

import json
import os
import random
import threading
from datetime import datetime, timezone

from rag import tracing
from rag.config import REPO_ROOT

LOG_DIR = REPO_ROOT / "logs"
ONLINE_JUDGE_RATE = float(os.getenv("ONLINE_JUDGE_RATE", "0.2"))
_lock = threading.Lock()


def _append(name: str, record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
    with _lock, (LOG_DIR / name).open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def log_request(request_id: str, answer) -> None:
    _append("requests.jsonl", {
        "request_id": request_id,
        "question_chars": len(answer.question),     # length only: questions may contain user data
        "answer_chars": len(answer.answer),
        "status": answer.status,
        "refusal_reason": answer.refusal_reason,
        "n_citations": len(answer.citations),
        "invalid_citations": answer.invalid_citations,
        "best_dense_score": max((hit["dense_score"] for hit in answer.retrieved), default=0.0),
        "latency_retrieval_ms": answer.latency_ms["retrieval"],
        "latency_generation_ms": answer.latency_ms["generation"],
        "latency_total_ms": answer.latency_ms["total"],
        "input_tokens": answer.tokens["input_tokens"],
        "output_tokens": answer.tokens["output_tokens"],
        "cost_usd": answer.cost_usd,
        "model": answer.model,
        "prompt_version": answer.prompt_version,
        "config_hash": answer.config_hash,
    })


def log_feedback(request_id: str, rating: str, comment: str | None) -> None:
    _append("feedback.jsonl", {"request_id": request_id, "rating": rating, "comment": comment})
    tracing.score(request_id, "user_feedback", 1.0 if rating == "up" else 0.0, comment)


def maybe_score_online(request_id: str, answer, context: str, judge) -> None:
    """Async LLM-judge on a sample of live answers: quality monitoring without a reference answer."""
    if judge is None or answer.status != "answered" or random.random() >= ONLINE_JUDGE_RATE:
        return
    try:
        score, _ = judge.faithfulness(answer.answer, context)
    except Exception as error:  # monitoring must never break serving; record the failure instead
        _append("online_scores.jsonl", {"request_id": request_id, "error": type(error).__name__})
        return
    tracing.score(request_id, "faithfulness_online", score)
    _append("online_scores.jsonl", {"request_id": request_id, "faithfulness": round(score, 4),
                                    "config_hash": answer.config_hash, "prompt_version": answer.prompt_version})
