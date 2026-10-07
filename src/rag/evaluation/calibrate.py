"""Judge calibration: how often does the LLM judge agree with a human?

  python -m rag.evaluation.calibrate export   # writes eval/calibration.md (to read) + eval/calibration.csv (to label)
  python -m rag.evaluation.calibrate score    # judge agreement with each filled label column

Label columns are kept apart and reported separately, never mixed:
  human_faithful   a person's label (the real calibration)
  panel_faithful   an independent AI panel (two opposing Claude labellers + adjudicator), a
                   cross-family check on the GPT judge, not a substitute for human labels

The review file hides the judge's verdict so the human label is independent. Label each answer
"yes" if every factual claim in it is supported by the excerpts shown, otherwise "no".
The judge counts as saying "yes" when its faithfulness score is 1.0 (every claim supported).
Answers come from the LLM response cache, so exporting costs nothing after an eval run.
"""

import csv
import json
import random
import sys

from rag.config import REPO_ROOT
from rag.evaluation.dataset import load_golden

RESULTS = REPO_ROOT / "eval" / "baseline.json"   # committed, so labels always match the same judge verdicts
GOLDEN = REPO_ROOT / "eval" / "golden.csv"
REVIEW = REPO_ROOT / "eval" / "calibration.md"
LABELS = REPO_ROOT / "eval" / "calibration.csv"
SAMPLE_SIZE = 20


def export() -> None:
    from rag.pipeline import RagPipeline

    rows = json.loads(RESULTS.read_text())["per_question"]["generation"]
    answered = [row for row in rows if row["status"] == "answered"]
    # Oversample answers the judge marked unfaithful, otherwise a 20-answer sample may contain none.
    flagged = [row for row in answered if row["faithfulness"] < 1.0]
    clean = [row for row in answered if row["faithfulness"] >= 1.0]
    rng = random.Random(3)
    sample = flagged[:SAMPLE_SIZE // 2] + rng.sample(clean, min(len(clean), SAMPLE_SIZE - min(len(flagged), SAMPLE_SIZE // 2)))
    rng.shuffle(sample)

    questions = {q.id: q.question for q in load_golden(GOLDEN)}
    pipeline = RagPipeline()
    lines = ["# Judge calibration: label these answers\n",
             "For each answer: is **every** factual claim supported by the excerpts above it? "
             "Write `yes` or `no` in the `human_faithful` column of `eval/calibration.csv`.\n"]
    for row in sample:
        result = pipeline.answer(questions[row["id"]], use_cache=True, include_context=True, trace_tags=("calibration",))
        lines += [f"---\n\n## {row['id']}\n", f"**Question:** {questions[row['id']]}\n",
                  "<details><summary>Excerpts the model saw</summary>\n", "```text", result.extras.get("context", ""), "```",
                  "</details>\n", f"**Answer:**\n\n{result.answer}\n"]
    REVIEW.write_text("\n".join(lines))
    with LABELS.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "human_faithful", "panel_faithful", "panel_how", "panel_reason", "note"])
        writer.writerows([[row["id"], "", "", "", "", ""] for row in sample])
    print(f"wrote {REVIEW.relative_to(REPO_ROOT)} and {LABELS.relative_to(REPO_ROOT)} ({len(sample)} answers)")


def _agreement(pairs: list[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    agree = sum(label == model for label, model in pairs) / n
    label_yes, judge_yes = sum(l for l, _ in pairs) / n, sum(m for _, m in pairs) / n
    expected = label_yes * judge_yes + (1 - label_yes) * (1 - judge_yes)
    kappa = (agree - expected) / (1 - expected) if expected < 1 else 1.0
    return {"labelled": n, "agreement": round(agree, 3), "cohens_kappa": round(kappa, 3),
            "labeller_says_faithful": round(label_yes, 3), "judge_says_faithful": round(judge_yes, 3)}


def score() -> None:
    judge = {row["id"]: row["faithfulness"] >= 1.0
             for row in json.loads(RESULTS.read_text())["per_question"]["generation"] if row["status"] == "answered"}
    with LABELS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = {"judge": "faithfulness == 1.0 means the judge says faithful"}
    for column in ("human_faithful", "panel_faithful"):
        labels = {row["id"]: (row.get(column) or "").strip().lower() for row in rows}
        usable = [qid for qid, value in labels.items() if value in ("yes", "no") and qid in judge]
        if not usable:
            continue
        summary[column] = {**_agreement([(labels[qid] == "yes", judge[qid]) for qid in usable]),
                           "disagreements": [qid for qid in usable if (labels[qid] == "yes") != judge[qid]]}
    if len(summary) == 1:
        sys.exit("No labels yet: fill human_faithful (or panel_faithful) with yes/no in eval/calibration.csv")
    (REPO_ROOT / "eval" / "calibration_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    {"export": export, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
