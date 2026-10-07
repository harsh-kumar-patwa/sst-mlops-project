"""Loads config/rag.yaml and derives the hashes used for versioning."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")  # API keys for local runs; CI and Render inject real env vars
CONFIG_PATH = REPO_ROOT / "config" / "rag.yaml"
PROMPTS_DIR = REPO_ROOT / "prompts"


def _short_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]


@lru_cache(maxsize=1)
def load_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    prompt_text = load_prompt(config["generation"]["prompt_version"])
    # The prompt text is part of the hash: editing a prompt file is a deployment event too.
    config["config_hash"] = _short_hash({"config": config, "prompt": prompt_text})
    # Only what changes the vectors decides whether the index must be rebuilt: the settings,
    # and the chunking/indexing code itself (a code-only change must not reuse a stale index).
    index_code = "".join((REPO_ROOT / "src" / "rag" / name).read_text() for name in ("chunking.py", "index.py"))
    config["index_hash"] = _short_hash({
        "corpus": config["corpus"], "chunking": config["chunking"],
        "embedding": config["embedding"], "code": index_code,
    })
    return config


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text()


def collection_name(config: dict) -> str:
    return f"vllm_docs__{config['index_hash']}"
