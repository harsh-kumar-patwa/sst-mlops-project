"""Monitoring dashboard over the service's request logs and the eval history.

Covers the five monitoring categories: operational (latency, cost, errors), input (question
length), output (refusals, citations), quality (online judge, thumbs up) and drift (retrieval
score and config version over time).
Run: make dashboard
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
sys.path.insert(0, str(ROOT / "src"))


def read_jsonl(name: str) -> pd.DataFrame:
    path = LOGS / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_json(path, lines=True)
    frame["ts"] = pd.to_datetime(frame["ts"])
    return frame


st.set_page_config(page_title="vLLM Docs RAG: Monitoring", layout="wide")
st.title("vLLM Docs RAG: monitoring")

requests, feedback, online = read_jsonl("requests.jsonl"), read_jsonl("feedback.jsonl"), read_jsonl("online_scores.jsonl")

if requests.empty:
    st.info("No traffic logged yet. Start the API with `make serve`, ask a few questions at http://localhost:8000, "
            "then refresh this page.")
else:
    answered = requests[requests["status"] == "answered"]
    latency = requests["latency_total_ms"]

    st.subheader("Operational")
    cols = st.columns(6)
    cols[0].metric("Requests", len(requests))
    cols[1].metric("p50 latency", f"{latency.quantile(0.50) / 1000:.2f} s")
    cols[2].metric("p95 latency", f"{latency.quantile(0.95) / 1000:.2f} s")
    cols[3].metric("p99 latency", f"{latency.quantile(0.99) / 1000:.2f} s")
    cols[4].metric("Cost / answered query", f"${answered['cost_usd'].mean():.5f}" if len(answered) else "–")
    cols[5].metric("LLM errors (degraded)", f"{(requests['status'] == 'degraded').mean():.1%}")

    left, right = st.columns(2)
    left.caption("Latency per request (ms), split by stage")
    left.area_chart(requests.set_index("ts")[["latency_retrieval_ms", "latency_generation_ms"]])
    right.caption("Cost per request (USD)")
    right.line_chart(requests.set_index("ts")[["cost_usd"]])

    st.subheader("Input and output")
    cols = st.columns(4)
    cols[0].metric("Median question length", f"{requests['question_chars'].median():.0f} chars")
    cols[1].metric("Refusal rate", f"{(requests['status'] == 'refused').mean():.1%}")
    cols[2].metric("Answers with valid citations",
                   f"{((answered['n_citations'] > 0) & (answered['invalid_citations'] == 0)).mean():.1%}" if len(answered) else "–")
    cols[3].metric("Median answer length", f"{answered['answer_chars'].median():.0f} chars" if len(answered) else "–")

    st.subheader("Quality")
    cols = st.columns(3)
    if not feedback.empty:
        cols[0].metric("Thumbs-up rate (north star)", f"{(feedback['rating'] == 'up').mean():.1%}",
                       help=f"{len(feedback)} ratings")
    else:
        cols[0].metric("Thumbs-up rate (north star)", "–")
    scored = online[online.get("faithfulness").notna()] if not online.empty and "faithfulness" in online else pd.DataFrame()
    cols[1].metric("Online faithfulness (sampled judge)", f"{scored['faithfulness'].mean():.3f}" if len(scored) else "–",
                   help="GPT-4o-mini scores a random sample of live answers")
    cols[2].metric("Judged answers", len(scored))

    st.subheader("Drift")
    left, right = st.columns(2)
    left.caption("Best retrieval score per request. A falling trend means a stale index or an embedding mismatch.")
    left.line_chart(requests.set_index("ts")["best_dense_score"].rolling(20, min_periods=1).median())
    right.caption("Traffic by config version (every prompt or config change is a deployment)")
    right.dataframe(requests.groupby(["config_hash", "prompt_version", "model"]).agg(
        requests=("request_id", "count"), p50_ms=("latency_total_ms", "median"), cost=("cost_usd", "mean"),
        first_seen=("ts", "min")).reset_index(), hide_index=True)

    st.subheader("Recent requests")
    st.dataframe(requests.sort_values("ts", ascending=False).head(50)[
        ["ts", "status", "latency_total_ms", "cost_usd", "n_citations", "best_dense_score", "config_hash"]], hide_index=True)

st.subheader("Offline eval: current baseline")
baseline_path = ROOT / "eval" / "baseline.json"
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())
    meta = baseline["meta"]
    st.caption(f"config {meta['config_hash']} · {meta['retrieval_mode']} · chunk {meta['chunk_size']} · "
               f"{meta['prompt_version']} · {meta['n_questions']} questions · {meta['timestamp']}")
    metrics = {**baseline["retrieval"]["overall"], **baseline.get("generation", {}).get("overall", {})}
    st.dataframe(pd.DataFrame([metrics]), hide_index=True)
    st.dataframe(pd.DataFrame(baseline["retrieval"]["by_slice"]).T, width="stretch")
