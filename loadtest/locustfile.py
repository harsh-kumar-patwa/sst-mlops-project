"""Load test.

By default only the retrieval path (/search) is exercised: it measures our own service's
throughput and costs nothing. Set LOADTEST_LLM=1 to also hit /ask, which calls Claude and
costs about $0.005 per request; keep users and duration small when you do.
"""

import csv
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.csv"
SMOKE = GOLDEN.with_name("smoke.csv")
with (GOLDEN if GOLDEN.exists() else SMOKE).open(newline="") as handle:
    QUESTIONS = [row["question"] for row in csv.DictReader(handle)]
CALL_LLM = os.getenv("LOADTEST_LLM") == "1"


class DocsUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(4)
    def search(self):
        self.client.get("/search", params={"q": random.choice(QUESTIONS)}, name="/search")

    @task(1)
    def ask(self):
        if CALL_LLM:
            self.client.post("/ask", json={"question": random.choice(QUESTIONS)}, name="/ask")
