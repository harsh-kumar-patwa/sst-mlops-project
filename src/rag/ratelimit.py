"""Request limits for a public deployment, where every /ask spends LLM credit.

Two limits, both in memory (one container, so no shared store is needed):
  per client   ASK_LIMIT_PER_MINUTE requests in any rolling 60 s window (default 10)
  whole site   ASK_LIMIT_PER_DAY requests per UTC day across all clients (default 300 ≈ $0.08)
A limit of 0 disables it.
"""

import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, clock=time.monotonic, today=None):
        self.per_minute, self.per_day = per_minute, per_day
        self.clock = clock
        self.today = today or (lambda: datetime.now(timezone.utc).date())
        self.recent: dict[str, deque] = defaultdict(deque)
        self.day, self.day_count = None, 0
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "RateLimiter":
        return cls(int(os.getenv("ASK_LIMIT_PER_MINUTE", "10")), int(os.getenv("ASK_LIMIT_PER_DAY", "300")))

    def check(self, client: str) -> str | None:
        """Records the request and returns None if allowed, or the reason it was refused."""
        now = self.clock()
        with self.lock:
            if self.day != self.today():
                self.day, self.day_count = self.today(), 0
            if self.per_day and self.day_count >= self.per_day:
                return "Daily question limit for this demo reached; please try again tomorrow."
            window = self.recent[client]
            while window and now - window[0] >= 60:
                window.popleft()
            if self.per_minute and len(window) >= self.per_minute:
                return f"Too many questions: limit is {self.per_minute} per minute. Please wait a moment."
            window.append(now)
            self.day_count += 1
            return None


def client_id(headers, fallback: str | None) -> str:
    # Behind the hosting proxy every request comes from the proxy's IP; the real client is the
    # first address in X-Forwarded-For.
    forwarded = headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or fallback or "unknown"
