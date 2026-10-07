---
title: vLLM Docs RAG
emoji: 📚
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Grounded Q&A over the vLLM docs, gated by evals in CI
---

# vLLM Docs RAG

Ask questions about the vLLM v0.31.0 documentation; answers cite their sources and the assistant says
"I don't know" when the docs don't cover the question.

This Space is deployed automatically from [harsh-kumar-patwa/sst-mlops-project](https://github.com/harsh-kumar-patwa/sst-mlops-project)
only after a change on `main` passes the evaluation gate. `GET /health` shows the config version being served.

Limits on this public demo: 10 questions per minute per visitor and 300 per day in total.
