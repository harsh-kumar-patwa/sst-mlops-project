"""On-disk cache for LLM calls, keyed by everything that affects the output.

Re-running an eval with an unchanged prompt, model and context costs nothing; any change to
those inputs misses the cache, so a regression can never be hidden by a stale cached answer.
Disabled when LLM_CACHE=off (the live API never uses it).
"""

import hashlib
import json
import os

from rag.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / ".cache" / "llm"


def _enabled() -> bool:
    return os.getenv("LLM_CACHE", "on") != "off"


def cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def get(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if _enabled() and path.exists():
        return json.loads(path.read_text())
    return None


def put(key: str, value: dict) -> None:
    if _enabled():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))
