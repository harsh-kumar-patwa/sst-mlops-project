"""The online question-answering path: retrieve -> guard -> generate -> validate citations.

Failure handling follows "degrade, never fail": if the LLM is down, rate-limited or slow,
the caller still gets the most relevant documentation excerpts instead of an error.
"""

import re
import time
from dataclasses import asdict, dataclass, field

from rag import llm_cache, tracing
from rag.llm import PROVIDER_ERRORS, ChatModel
from rag.config import load_config, load_prompt
from rag.retrieve import Hit, Retriever

REFUSAL = "I don't know based on the vLLM documentation."
CITATION = re.compile(r"\[(\d+)\]")
CHARS_PER_TOKEN = 4


@dataclass
class Answer:
    question: str
    answer: str
    status: str                       # answered | refused | degraded
    refusal_reason: str | None
    citations: list[dict]
    retrieved: list[dict]
    invalid_citations: int            # [n] markers that point to no retrieved excerpt
    latency_ms: dict
    tokens: dict
    cost_usd: float
    model: str
    prompt_version: str
    config_hash: str
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RagPipeline:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.model = ChatModel(self.config["generation"])
        self.retriever = Retriever(self.config)
        self.system_prompt = load_prompt(self.config["generation"]["prompt_version"])

    def build_context(self, hits: list[Hit]) -> tuple[str, list[Hit]]:
        budget = self.config["generation"]["max_context_tokens"] * CHARS_PER_TOKEN
        blocks, used = [], []
        for hit in hits:
            block = (f"[BEGIN DOCUMENT {len(used) + 1}] source: {hit.doc_path} | section: {hit.breadcrumb}\n"
                     f"{hit.text}\n[END DOCUMENT {len(used) + 1}]")
            if used and sum(map(len, blocks)) + len(block) > budget:
                break
            blocks.append(block)
            used.append(hit)
        return "\n\n".join(blocks), used

    def generate(self, question: str, context: str, use_cache: bool) -> dict:
        request = self.model.request(self.system_prompt,
                                     f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}")
        key = llm_cache.cache_key(request)
        if use_cache and (cached := llm_cache.get(key)):
            return {**cached, "cached": True}
        result = self.model.complete(request)
        if use_cache:
            llm_cache.put(key, result)
        return {**result, "cached": False}

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        price = self.config["generation"]["price_per_mtok"]
        return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    def answer(self, question: str, use_cache: bool = False, include_context: bool = False,
               trace_tags: tuple[str, ...] = ("serving",)) -> Answer:
        generation = self.config["generation"]
        with tracing.trace(
            "rag-answer", input=question, version=self.config["config_hash"],
            tags=[*trace_tags, generation["prompt_version"], self.config["retrieval"]["mode"]],
            metadata={"config_hash": self.config["config_hash"], "index_hash": self.config["index_hash"],
                      "prompt_version": generation["prompt_version"], "model": generation["model"]},
        ) as (root, trace_id):
            result = self._answer(question, use_cache, include_context)
            root.update(output=result.answer, metadata={"status": result.status, "reason": result.refusal_reason,
                                                        "cost_usd": result.cost_usd, "latency_ms": result.latency_ms})
        result.extras["trace_id"] = trace_id
        return result

    def _answer(self, question: str, use_cache: bool, include_context: bool) -> Answer:
        started = time.perf_counter()
        with tracing.span("retrieve", as_type="retriever", input=question) as span:
            hits = self.retriever.search(question)
            span.update(output=[{"doc_path": h.doc_path, "section": h.breadcrumb, "score": round(h.score, 4)}
                                for h in hits])
        retrieval_ms = (time.perf_counter() - started) * 1000
        retrieved = [{"rank": i + 1, "chunk_id": h.chunk_id, "doc_path": h.doc_path, "section": h.breadcrumb,
                      "score": round(h.score, 4), "dense_score": round(h.dense_score, 4)} for i, h in enumerate(hits)]

        def finish(text, status, reason=None, citations=(), invalid=0, generation_ms=0.0, usage=None, extras=None):
            usage = usage or {"input_tokens": 0, "output_tokens": 0}
            return Answer(
                question=question, answer=text, status=status, refusal_reason=reason,
                citations=list(citations), retrieved=retrieved, invalid_citations=invalid,
                latency_ms={"retrieval": round(retrieval_ms, 1), "generation": round(generation_ms, 1),
                            "total": round((time.perf_counter() - started) * 1000, 1)},
                tokens=usage, cost_usd=round(self.cost(usage["input_tokens"], usage["output_tokens"]), 6),
                model=self.config["generation"]["model"], prompt_version=self.config["generation"]["prompt_version"],
                config_hash=self.config["config_hash"], extras=extras or {},
            )

        best_score = max((h.dense_score for h in hits), default=0.0)
        if best_score < self.config["retrieval"]["min_score"]:
            return finish(REFUSAL, "refused", reason=f"best retrieval score {best_score:.2f} below threshold")

        context, used = self.build_context(hits)
        generation_started = time.perf_counter()
        with tracing.span("generate", as_type="generation", model=self.config["generation"]["model"],
                          input={"system": self.system_prompt, "context": context, "question": question},
                          metadata={"prompt_version": self.config["generation"]["prompt_version"]}) as span:
            try:
                result = self.generate(question, context, use_cache)
            except PROVIDER_ERRORS as error:
                span.update(level="ERROR", status_message=type(error).__name__)
                fallback = "Summary unavailable right now. The most relevant documentation sections are:\n" + \
                    "\n".join(f"- {h.breadcrumb} ({h.doc_path})" for h in used)
                return finish(fallback, "degraded", reason=f"{type(error).__name__}",
                              generation_ms=(time.perf_counter() - generation_started) * 1000)
            span.update(output=result["text"],
                        usage_details={"input": result["input_tokens"], "output": result["output_tokens"]},
                        cost_details={"total": self.cost(result["input_tokens"], result["output_tokens"])},
                        metadata={"cached": result["cached"], "stop_reason": result["stop_reason"]})
        generation_ms = (time.perf_counter() - generation_started) * 1000

        text = result["text"]
        usage = {"input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"]}
        extras = {"cached": result["cached"], "stop_reason": result["stop_reason"]}
        if include_context:
            extras["context"] = context
        if text.startswith(REFUSAL):
            return finish(REFUSAL, "refused", reason="model found no answer in the excerpts",
                          generation_ms=generation_ms, usage=usage, extras=extras)

        cited_numbers = sorted({int(n) for n in CITATION.findall(text)})
        citations = [{"n": n, "doc_path": used[n - 1].doc_path, "section": used[n - 1].breadcrumb,
                      "chunk_id": used[n - 1].chunk_id} for n in cited_numbers if 1 <= n <= len(used)]
        invalid = sum(1 for n in cited_numbers if not 1 <= n <= len(used))
        return finish(text, "answered", citations=citations, invalid=invalid, generation_ms=generation_ms,
                      usage=usage, extras=extras)
