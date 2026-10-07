# Golden set: how to write `golden.csv`

The golden set is the answer key the CI gate grades every change against. It must be
**hand-written** from the vLLM v0.31.0 docs in `data/vllm-docs/` (browse them on GitHub or
at docs.vllm.ai). Target: **60 questions**.

`eval/smoke.csv` is only a 6-question smoke test used until `golden.csv` exists. It is not the golden set.

## Columns

| Column | What to write |
|---|---|
| `id` | Unique, e.g. `q001` |
| `question` | How a real user would ask it. Don't copy the docs' wording, paraphrase. |
| `reference_answer` | The correct answer in 1–2 sentences, with exact flag/parameter names |
| `source_docs` | Doc path(s) that contain the answer, exactly as listed in `doc_map.md`. Several paths separated by `;` |
| `source_section` | Section heading where the answer is (optional, used for the Section@5 metric) |
| `slice` | One of `factual`, `how_to`, `config_flag`, `multi_hop`, `unanswerable` |
| `author` | Who wrote it |

Leave `source_docs` empty for `unanswerable` questions.

## Mix for 60 questions

| Slice | Count | Example |
|---|---|---|
| `factual` | 15 | What port does the OpenAI-compatible server use by default? |
| `how_to` | 15 | How do I serve a LoRA adapter alongside the base model? |
| `config_flag` | 12 | Which setting limits how much GPU memory vLLM pre-allocates? |
| `multi_hop` | 8 | The answer needs two pages; list both in `source_docs` (both must be retrieved) |
| `unanswerable` | 10 | Plausible but not in the docs: pricing, unreleased features, other products |

## Rules

- Every answerable question must be answerable **from the docs alone**.
- For non-multi-hop questions, list every page that fully answers it. Any one of them counts as a correct retrieval.
- Spread questions across sections: getting_started, serving, configuration, features, deployment, models.
- Hold-out: keep the last 15 rows (`q046`–`q060`) for final reporting and don't tune prompts or chunking against them.
- Run `make eval-retrieval` after adding questions. The loader rejects unknown slices and doc paths that don't exist.
