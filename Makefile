PY      ?= .venv/bin/python
GOLDEN  ?= $(if $(wildcard eval/golden.csv),eval/golden.csv,eval/smoke.csv)
export PYTHONPATH := src

.PHONY: setup index test eval eval-retrieval gate baseline serve dashboard loadtest sweep docmap

setup:            ## create the virtualenv and install everything
	python3.12 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-dev.txt

index:            ## chunk + embed the vLLM docs into Qdrant (skips if this index version exists)
	$(PY) -m rag.index

test:             ## unit tests
	$(PY) -m pytest -q

eval:             ## full eval: retrieval + Claude answers + GPT-4o-mini judge (needs API keys)
	$(PY) -m rag.evaluation.run --golden $(GOLDEN)

eval-retrieval:   ## retrieval-only eval: free, deterministic, no keys
	$(PY) -m rag.evaluation.run --golden $(GOLDEN) --retrieval-only

gate:             ## compare the latest results with the baseline, the same check CI runs
	$(PY) -m rag.evaluation.compare eval/baseline.json eval/results/latest.json

baseline:         ## accept the latest results as the new baseline (commit it in the same PR)
	cp eval/results/latest.json eval/baseline.json

serve:            ## run the API + chat page on http://localhost:8000
	$(PY) -m uvicorn rag.api:app --app-dir src --port 8000

dashboard:        ## monitoring dashboard on http://localhost:8501
	$(PY) -m streamlit run dashboard/app.py

loadtest:         ## 60 s load test against a running server (retrieval-only by default, costs nothing)
	$(PY) -m locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 60s --csv loadtest/results

sweep:            ## experiment: chunk size x retrieval mode, retrieval metrics only
	$(PY) -m rag.evaluation.sweep --golden $(GOLDEN)

docmap:           ## list every doc page and section heading, to help write golden questions
	$(PY) -m rag.evaluation.docmap > eval/doc_map.md
