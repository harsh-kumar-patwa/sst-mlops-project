"""The CI eval gate: compares a results file against the baseline using config/gate.yaml.

Usage: python -m rag.evaluation.compare eval/baseline.json eval/results/latest.json [--report report.md]
Exits 1 if any rule fails, which fails the GitHub Actions check and blocks the merge.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from rag.config import REPO_ROOT

GATE_PATH = REPO_ROOT / "config" / "gate.yaml"


def _check(name: str, base: float | None, current: float | None, rule: dict) -> dict:
    if current is None:
        return {"metric": name, "baseline": base, "current": None, "delta": None, "ok": True, "rule": "not measured"}
    delta = None if base is None else round(current - base, 4)
    failures = []
    if "min" in rule and current < rule["min"]:
        failures.append(f"below floor {rule['min']}")
    if "max_drop" in rule and delta is not None and delta < -rule["max_drop"]:
        failures.append(f"dropped more than {rule['max_drop']}")
    rule_text = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in rule.items())
    return {"metric": name, "baseline": base, "current": current, "delta": delta,
            "ok": not failures, "rule": "; ".join(failures) or rule_text}


def same_question_set(baseline: dict, current: dict) -> bool:
    keys = ("golden", "n_questions")
    return all(baseline.get("meta", {}).get(k) == current.get("meta", {}).get(k) for k in keys)


def compare(baseline: dict, current: dict, gate: dict) -> list[dict]:
    if not same_question_set(baseline, current):
        # Deltas between different question sets mean nothing; only the absolute floors apply.
        baseline = {"meta": baseline.get("meta", {}), "retrieval": {"overall": {}, "by_slice": {}}}
    checks = []
    for metric, rule in gate["retrieval"].items():
        if metric == "slice_max_drop":
            continue
        checks.append(_check(metric, baseline["retrieval"]["overall"].get(metric),
                             current["retrieval"]["overall"].get(metric), rule))
    for name, values in current["retrieval"]["by_slice"].items():
        base = baseline["retrieval"]["by_slice"].get(name, {}).get("recall_at_5")
        checks.append(_check(f"recall_at_5 [{name}]", base, values["recall_at_5"],
                             {"max_drop": gate["retrieval"]["slice_max_drop"]}))
    if "generation" in current:
        base_generation = baseline.get("generation", {}).get("overall", {})
        for metric, rule in gate["generation"].items():
            checks.append(_check(metric, base_generation.get(metric), current["generation"]["overall"].get(metric), rule))
        if current["generation"]["overall"]["degraded"]:
            checks.append({"metric": "degraded answers", "baseline": 0, "ok": False, "delta": None, "rule": "LLM errors during eval",
                           "current": current["generation"]["overall"]["degraded"]})
    return checks


def render_report(checks: list[dict], baseline: dict, current: dict) -> str:
    def fmt(value):
        if value is None:
            return "–"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    passed = all(check["ok"] for check in checks)
    meta, base_meta = current["meta"], baseline["meta"]
    lines = [
        f"## {'✅ Eval gate passed' if passed else '❌ Eval gate failed: this PR lowers answer quality'}",
        "",
        f"Comparing config `{meta['config_hash']}` ({meta['retrieval_mode']}, chunk {meta['chunk_size']}, "
        f"{meta['prompt_version']}) against baseline `{base_meta['config_hash']}` "
        f"on {meta['n_questions']} golden questions.",
        "",
        *([] if same_question_set(baseline, current) else [
            "> ⚠️ The golden set changed in this PR, so only the absolute floors are checked. "
            "Commit a new `eval/baseline.json` with this PR (`make eval && make baseline`).", ""]),
        "| Metric | Baseline | This PR | Δ | Result |",
        "|---|---|---|---|---|",
    ]
    for check in checks:
        delta = "–" if check["delta"] is None else f"{check['delta']:+.3f}"
        status = "✅ pass" if check["ok"] else f"❌ {check['rule']}"
        lines.append(f"| {check['metric']} | {fmt(check['baseline'])} | {fmt(check['current'])} | {delta} | {status} |")
    if "generation" in current:
        cost, latency = current["generation"]["cost"], current["generation"]["latency_ms"]
        lines += ["", f"Cost per query: ${cost['generation_per_query_usd']:.5f} · judge spend this run: "
                      f"${cost['judge_spent_this_run_usd']:.4f} · latency p50/p99: {latency['p50']}/{latency['p99']} ms "
                      f"({latency['n_uncached']} uncached calls)"]
    lines += ["", "<sub>Rules live in `config/gate.yaml`. Retrieval metrics are deterministic; "
                  "judge metrics (GPT-4o-mini) have wider tolerances.</sub>"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--report", type=Path, help="write the markdown report here (for the PR comment)")
    args = parser.parse_args()

    baseline, current = json.loads(args.baseline.read_text()), json.loads(args.current.read_text())
    checks = compare(baseline, current, yaml.safe_load(GATE_PATH.read_text()))
    report = render_report(checks, baseline, current)
    print(report)
    if args.report:
        args.report.write_text(report)
    sys.exit(0 if all(check["ok"] for check in checks) else 1)


if __name__ == "__main__":
    main()
