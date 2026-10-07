"""FastAPI service.

POST /ask        question -> grounded answer with citations, cost, latency and version stamps
POST /feedback   thumbs up/down on an answer (online north-star metric)
GET  /search     retrieval only, no LLM call (debugging and cost-free load testing)
GET  /health     liveness plus the config and index versions being served
GET  /           minimal chat page for demos

Run: uvicorn rag.api:app --app-dir src
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag import telemetry, tracing
from rag.pipeline import RagPipeline

STATIC_PAGE = Path(__file__).with_name("chat.html")
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["pipeline"] = RagPipeline()
    state["judge"] = None
    if os.getenv("OPENAI_API_KEY"):
        from rag.evaluation.judge import Judge
        state["judge"] = Judge(state["pipeline"].config)
    yield
    tracing.flush()
    state.clear()


app = FastAPI(title="vLLM Docs RAG", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


# Sync handlers: FastAPI runs them in a worker thread, so a slow LLM call never blocks the event loop.
@app.post("/ask")
def ask(body: AskRequest, background: BackgroundTasks) -> dict:
    answer = state["pipeline"].answer(body.question, include_context=True)
    # With tracing on, the request id IS the Langfuse trace id, so feedback attaches to the right trace.
    request_id = answer.extras.get("trace_id") or uuid.uuid4().hex
    context = answer.extras.pop("context", "")
    background.add_task(telemetry.log_request, request_id, answer)
    background.add_task(telemetry.maybe_score_online, request_id, answer, context, state["judge"])
    return {"request_id": request_id, **answer.to_dict()}


@app.post("/feedback")
def feedback(body: FeedbackRequest, background: BackgroundTasks) -> dict:
    background.add_task(telemetry.log_feedback, body.request_id, body.rating, body.comment)
    return {"ok": True}


@app.get("/search")
def search(q: str, top_k: int = 5) -> dict:
    if not 1 <= top_k <= 20:
        raise HTTPException(422, "top_k must be between 1 and 20")
    hits = state["pipeline"].retriever.search(q, top_k=top_k)
    return {"query": q, "hits": [{"doc_path": h.doc_path, "section": h.breadcrumb, "score": round(h.score, 4),
                                  "text": h.text} for h in hits]}


@app.get("/health")
def health() -> dict:
    pipeline = state["pipeline"]
    config = pipeline.config
    return {
        "status": "ok",
        "config_hash": config["config_hash"],
        "index": pipeline.retriever.collection,
        "chunks": pipeline.retriever.client.count(pipeline.retriever.collection).count,
        "corpus_version": config["corpus"]["version"],
        "retrieval_mode": config["retrieval"]["mode"],
        "generator": config["generation"]["model"],
        "prompt_version": config["generation"]["prompt_version"],
        "online_judge": state["judge"] is not None,
        "langfuse_tracing": tracing.enabled(),
    }


@app.get("/", response_class=HTMLResponse)
def chat_page() -> str:
    return STATIC_PAGE.read_text()
