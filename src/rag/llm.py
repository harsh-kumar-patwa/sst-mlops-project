"""One interface over the chat-model providers, selected by `provider` in config/rag.yaml.

Swapping the generator (e.g. OpenAI <-> Anthropic) is a config change, so it goes through the
same eval gate as any other change.
"""

import os

import anthropic
import openai

API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

# Errors that mean "the provider is unavailable right now" and should degrade, not crash.
PROVIDER_ERRORS = (
    openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError,
    anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError,
)


class ChatModel:
    def __init__(self, settings: dict):
        self.provider = settings["provider"]
        if self.provider not in API_KEY_VARS:
            raise ValueError(f"unknown provider '{self.provider}', expected one of {sorted(API_KEY_VARS)}")
        if not os.getenv(API_KEY_VARS[self.provider]):
            raise RuntimeError(f"{API_KEY_VARS[self.provider]} is not set. Copy .env.example to .env and fill it in.")
        self.settings = settings
        timeout = settings.get("timeout_seconds", 60)
        self.client = (openai.OpenAI(timeout=timeout, max_retries=2) if self.provider == "openai"
                       else anthropic.Anthropic(timeout=timeout, max_retries=2))

    def request(self, system: str, user: str) -> dict:
        """The full request payload; also used as the cache key, so any change misses the cache."""
        settings = self.settings
        if self.provider == "openai":
            return {"model": settings["model"], "temperature": settings["temperature"], "seed": 7,
                    "max_completion_tokens": settings["max_tokens"],
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        # anthropic 1.x dropped the temperature keyword; Haiku 4.5 still honours it and the eval relies
        # on temperature 0 for repeatable answers, so it goes in the raw request body.
        return {"model": settings["model"], "max_tokens": settings["max_tokens"], "system": system,
                "extra_body": {"temperature": settings["temperature"]},
                "messages": [{"role": "user", "content": user}]}

    def complete(self, request: dict) -> dict:
        if self.provider == "openai":
            response = self.client.chat.completions.create(**request)
            return {"text": (response.choices[0].message.content or "").strip(),
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "stop_reason": response.choices[0].finish_reason}
        response = self.client.messages.create(**request)
        return {"text": "".join(block.text for block in response.content if block.type == "text").strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason}
