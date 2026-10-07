from datetime import date

from rag.ratelimit import RateLimiter, client_id


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_per_client_window_rolls_over():
    clock = Clock()
    limiter = RateLimiter(per_minute=2, per_day=0, clock=clock)
    assert limiter.check("a") is None and limiter.check("a") is None
    assert "per minute" in limiter.check("a")
    assert limiter.check("b") is None          # other clients are unaffected
    clock.now = 61
    assert limiter.check("a") is None          # window has rolled over


def test_daily_cap_is_global_and_resets_each_day():
    day = {"value": date(2026, 10, 9)}
    limiter = RateLimiter(per_minute=0, per_day=2, today=lambda: day["value"])
    assert limiter.check("a") is None and limiter.check("b") is None
    assert "Daily" in limiter.check("c")
    day["value"] = date(2026, 10, 10)
    assert limiter.check("c") is None


def test_client_id_prefers_forwarded_header():
    assert client_id({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.1") == "1.2.3.4"
    assert client_id({}, "10.0.0.1") == "10.0.0.1"
