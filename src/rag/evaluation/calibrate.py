"""Judge calibration: how often does the LLM judge agree with a human?

  python -m rag.evaluation.calibrate export   # writes eval/calibration.md (to read) + eval/calibration.csv (to label)
  python -m rag.evaluation.calibrate score    # agreement between your labels and the judge

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
        writer.writerow(["id", "human_faithful", "note"])
        writer.writerows([[row["id"], "", ""] for row in sample])
    print(f"wrote {REVIEW.relative_to(REPO_ROOT)} and {LABELS.relative_to(REPO_ROOT)} ({len(sample)} answers)")


def score() -> None:
    judge = {row["id"]: row["faithfulness"] >= 1.0
             for row in json.loads(RESULTS.read_text())["per_question"]["generation"] if row["status"] == "answered"}
    with LABELS.open(newline="") as handle:
        labels = {row["id"]: row["human_faithful"].strip().lower() for row in csv.DictReader(handle)}
    pairs = [(labels[qid] == "yes", judge[qid]) for qid in labels if labels[qid] in ("yes", "no") and qid in judge]
    if not pairs:
        sys.exit("No labels yet: fill human_faithful with yes/no in eval/calibration.csv")

    n = len(pairs)
    agree = sum(human == model for human, model in pairs) / n
    human_yes, judge_yes = sum(h for h, _ in pairs) / n, sum(m for _, m in pairs) / n
    expected = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)
    kappa = (agree - expected) / (1 - expected) if expected < 1 else 1.0
    disagreements = [qid for qid in labels if qid in judge and labels[qid] in ("yes", "no")
                     and (labels[qid] == "yes") != judge[qid]]
    summary = {"labelled": n, "agreement": round(agree, 3), "cohens_kappa": round(kappa, 3),
               "human_says_faithful": round(human_yes, 3), "judge_says_faithful": round(judge_yes, 3),
               "disagreements": disagreements}
    (REPO_ROOT / "eval" / "calibration_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    {"export": export, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
