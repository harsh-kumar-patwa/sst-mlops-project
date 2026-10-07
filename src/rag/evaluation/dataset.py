"""Loads and validates the hand-written golden set (eval/golden.csv)."""

import csv
from dataclasses import dataclass
from pathlib import Path

from rag.config import REPO_ROOT, load_config

SLICES = {"factual", "how_to", "config_flag", "multi_hop", "unanswerable"}


@dataclass
class GoldenQuestion:
    id: str
    question: str
    reference_answer: str
    source_docs: list[str]      # any listed doc counts as relevant; multi_hop needs all of them
    source_section: str
    slice: str
    author: str

    @property
    def answerable(self) -> bool:
        return self.slice != "unanswerable"


def load_golden(path: Path) -> list[GoldenQuestion]:
    corpus_dir = REPO_ROOT / load_config()["corpus"]["path"]
    questions, problems = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            docs = [doc.strip() for doc in (row.get("source_docs") or "").split(";") if doc.strip()]
            question = GoldenQuestion(
                id=row["id"].strip(), question=row["question"].strip(),
                reference_answer=(row.get("reference_answer") or "").strip(), source_docs=docs,
                source_section=(row.get("source_section") or "").strip(),
                slice=row["slice"].strip(), author=(row.get("author") or "").strip(),
            )
            if question.slice not in SLICES:
                problems.append(f"line {line_number}: unknown slice '{question.slice}'")
            if question.answerable and not docs:
                problems.append(f"line {line_number}: answerable question needs source_docs")
            problems += [f"line {line_number}: '{doc}' is not a file in {corpus_dir.name}"
                         for doc in docs if not (corpus_dir / doc).exists()]
            questions.append(question)

    duplicate_ids = {q.id for q in questions if [x.id for x in questions].count(q.id) > 1}
    problems += [f"duplicate id '{qid}'" for qid in sorted(duplicate_ids)]
    if problems:
        raise ValueError(f"{path} has {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return questions
