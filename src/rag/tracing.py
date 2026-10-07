"""Langfuse tracing. Every answer becomes one trace with a span per stage:

  rag-answer (chain)            question in, answer + status out; tagged with config_hash
    ├── retrieve (retriever)    top-k pages and scores
    └── generate (generation)   model, prompt version, tokens, cost

Spans are exported by a background thread (OpenTelemetry batching), so tracing adds no latency
to the request. If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, every call here is a
no-op and the service runs exactly the same.
"""

import os
from contextlib import contextmanager, nullcontext
from functools import lru_cache


class _NoopObservation:
    def update(self, **_):
        return self


@lru_cache(maxsize=1)
def client():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    from langfuse import Langfuse
    return Langfuse(environment=os.getenv("LANGFUSE_ENV", "development"))


def enabled() -> bool:
    return client() is not None


@contextmanager
def trace(name: str, *, tags: list[str], version: str, metadata: dict, input: object):
    """Root observation; tags and version propagate to every child span in the trace."""
    langfuse = client()
    if langfuse is None:
        yield _NoopObservation(), None
        return
    from langfuse import propagate_attributes
    with propagate_attributes(tags=tags, version=version, metadata=metadata, trace_name=name):
        with langfuse.start_as_current_observation(name=name, as_type="chain", input=input) as root:
            yield root, langfuse.get_current_trace_id()


def span(name: str, as_type: str = "span", **fields):
    langfuse = client()
    if langfuse is None:
        return nullcontext(_NoopObservation())
    return langfuse.start_as_current_observation(name=name, as_type=as_type, **fields)


def score(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    """Attach a quality score (online judge, user feedback) to an existing trace."""
    langfuse = client()
    if langfuse is not None and trace_id:
        langfuse.create_score(trace_id=trace_id, name=name, value=value, comment=comment)


def flush() -> None:
    if (langfuse := client()) is not None:
        langfuse.flush()
