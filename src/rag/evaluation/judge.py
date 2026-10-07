"""LLM-as-judge scoring with a model from a different family than the generator.

faithfulness  share of the answer's factual claims that the retrieved excerpts support
              (catches hallucination; reference-free, so it also works on live traffic).
correctness   does the answer agree with the hand-written reference answer: 1, 0.5 or 0.
Judge calls are cached on disk like generator calls, so re-scoring unchanged answers is free.
"""

import json
import os

from openai import OpenAI

from rag import llm_cache

FAITHFULNESS_PROMPT = """You check whether an answer is grounded in documentation excerpts.
Split the ANSWER into atomic factual claims (ignore citation markers like [1]).
For each claim decide if the EXCERPTS directly support it. Paraphrase counts as support;
claims needing outside knowledge do not.
Return JSON: {"claims": [{"claim": "...", "supported": true|false}]}

EXCERPTS:
{context}

ANSWER:
{answer}"""

CORRECTNESS_PROMPT = """You grade an answer to a question about the vLLM documentation against a reference answer
written by a human expert.
Score 1 if the answer contains the key facts of the reference and contradicts nothing in it,
0.5 if it is partially right or misses an important part, 0 if it is wrong, off-topic or refuses.
Extra correct detail is fine.
Return JSON: {"score": 1|0.5|0, "reason": "one sentence"}

QUESTION: {question}
REFERENCE ANSWER: {reference}
ANSWER: {answer}"""


class Judge:
    def __init__(self, config: dict):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        self.settings = config["judge"]
        self.client = OpenAI(timeout=60, max_retries=3)
        self.cost_usd = 0.0

    def _ask(self, prompt: str) -> dict:
        request = {
            "model": self.settings["model"],
            "temperature": self.settings["temperature"],
            "seed": 7,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        key = llm_cache.cache_key({"judge": request})
        if cached := llm_cache.get(key):
            return cached
        response = self.client.chat.completions.create(**request)
        price = self.settings["price_per_mtok"]
        self.cost_usd += (response.usage.prompt_tokens * price["input"]
                          + response.usage.completion_tokens * price["output"]) / 1_000_000
        result = json.loads(response.choices[0].message.content)
        llm_cache.put(key, result)
        return result

    def faithfulness(self, answer: str, context: str) -> tuple[float, list[dict]]:
        # str.replace, not str.format: excerpts contain literal braces (JSON, Python dicts).
        prompt = FAITHFULNESS_PROMPT.replace("{context}", context).replace("{answer}", answer)
        claims = self._ask(prompt).get("claims", [])
        if not claims:
            return 1.0, []
        return sum(1 for claim in claims if claim.get("supported")) / len(claims), claims

    def correctness(self, question: str, reference: str, answer: str) -> tuple[float, str]:
        prompt = (CORRECTNESS_PROMPT.replace("{question}", question)
                  .replace("{reference}", reference).replace("{answer}", answer))
        verdict = self._ask(prompt)
        score = float(verdict.get("score", 0))
        return (score if score in (0.0, 0.5, 1.0) else 0.0), verdict.get("reason", "")
