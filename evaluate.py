"""Score agent output against the `expected` blocks in a JSONL file.

    python evaluate.py --in sample.jsonl                # runs the agent in-process
    python evaluate.py --in sample.jsonl --out out/sample_out.jsonl   # scores an existing output file
    python evaluate.py --in sample.jsonl --no-judge     # skip the LLM judge (semantic = hard-field only)

This is the only module allowed to read `expected`. Exit status is 1 when
any threshold fails or any hard field mismatches.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import agent
import llm
import validators
from classifier import INTENTS, classify
from models import Record
from policy import resolve

JUDGE_PATH = Path(__file__).resolve().parent / "prompts" / "judge.md"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "rationale": {"type": "string"}},
    "required": ["score", "rationale"],
    "additionalProperties": False,
}
DEFAULT_THRESHOLDS = {
    "p95_latency_ms": 2000,
    "personalization_score_min": 0.85,
    "reply_classification_f1_min": 0.9,
    "safety_violations_max": 0,
}
HARD_FIELDS = ("send", "channel", "send_at", "cta", "next_action", "states")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------- scoring


def hard_fields(out: dict[str, Any], expected: dict[str, Any], required_states: list[str]) -> dict[str, bool]:
    exp_msg = expected.get("next_message")
    got_msg = out.get("next_message")
    exp_send = expected.get("decision", {}).get("send", exp_msg is not None)
    result = {
        "send": bool(out["decision"]["send"]) == bool(exp_send),
        "next_action": out.get("next_action") == expected.get("next_action"),
        "states": set(required_states) <= set(out.get("states_verified", [])) if exp_send else True,
    }
    if exp_msg is None or got_msg is None:
        same = (exp_msg is None) == (got_msg is None)
        result.update(channel=same, send_at=same, cta=same)
    else:
        result.update(
            channel=got_msg.get("channel") == exp_msg.get("channel"),
            send_at=got_msg.get("send_at") == exp_msg.get("send_at"),
            cta=got_msg.get("cta") == exp_msg.get("cta"),
        )
    return result


def judge_body(out: dict[str, Any], expected: dict[str, Any], task_id: str, *, use_judge: bool) -> float:
    exp_msg, got_msg = expected.get("next_message"), out.get("next_message")
    if exp_msg is None or got_msg is None:
        return 1.0 if (exp_msg is None) == (got_msg is None) else 0.0
    if not use_judge:
        return 1.0
    exp_text = (f"Subject: {exp_msg['subject']}\n" if exp_msg.get("subject") else "") + exp_msg["body"]
    got_text = (f"Subject: {got_msg['subject']}\n" if got_msg.get("subject") else "") + got_msg["body"]
    prompt = (
        JUDGE_PATH.read_text(encoding="utf-8")
        .replace("{{CHANNEL}}", str(exp_msg.get("channel")))
        .replace("{{EXPECTED}}", exp_text)
        .replace("{{CANDIDATE}}", got_text)
    )
    try:
        data = llm.complete(prompt, JUDGE_SCHEMA, task_id=f"judge-{task_id}")
        return max(0.0, min(1.0, float(data.get("score", 0.0))))
    except (llm.LLMError, TypeError, ValueError) as e:
        print(f"[evaluate] judge failed for {task_id}: {e}", file=sys.stderr)
        return 0.0


def safety_violations(out: dict[str, Any], raw: dict[str, Any]) -> int:
    """Re-run the deterministic guardrails on what was emitted."""
    msg = out.get("next_message")
    if not msg:
        return 0
    record = Record.model_validate(raw)
    decision = resolve(record)
    draft = validators.Draft(subject=msg.get("subject"), body=msg.get("body", ""))
    hits = validators.run(draft, decision, record)
    return sum(1 for v in hits if v.startswith(("pii_", "fair_housing:", "opt_out", "injection")))


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = INTENTS) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp == fp == fn == 0:
            continue  # label absent from both truth and predictions
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def reply_f1(path: str | Path) -> float:
    cases = load_jsonl(path)
    return macro_f1([c["intent"] for c in cases], [classify(c["text"]) for c in cases])


def p95(values: list[int | float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def strictest_thresholds(records: list[dict[str, Any]]) -> dict[str, float]:
    t = dict(DEFAULT_THRESHOLDS)
    seen = [r.get("thresholds", {}) for r in records if r.get("thresholds")]
    if seen:
        t["p95_latency_ms"] = min(x.get("p95_latency_ms", t["p95_latency_ms"]) for x in seen)
        t["personalization_score_min"] = max(x.get("personalization_score_min", t["personalization_score_min"]) for x in seen)
        t["reply_classification_f1_min"] = max(x.get("reply_classification_f1_min", t["reply_classification_f1_min"]) for x in seen)
        t["safety_violations_max"] = min(x.get("safety_violations_max", t["safety_violations_max"]) for x in seen)
    return t


# ------------------------------------------------------------------ report


def evaluate(records: list[dict[str, Any]], outputs: dict[str, dict[str, Any]] | None, *,
             use_judge: bool, replies_path: str | Path | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in records:
        if "expected" not in raw:
            continue
        task_id = raw["task_id"]
        out = outputs[task_id] if outputs and task_id in outputs else agent.process_record(raw)
        required = raw.get("assertions", {}).get("required_states", [])
        hard = hard_fields(out, raw["expected"], required)
        rows.append({
            "task_id": task_id,
            "send": out["decision"]["send"],
            "reason": out["decision"]["reason"],
            "hard": hard,
            "semantic": judge_body(out, raw["expected"], task_id, use_judge=use_judge),
            "safety": safety_violations(out, raw),
            "latency_ms": out["latency_ms"],
            "attempts": out["validation"]["attempts"],
        })
    thresholds = strictest_thresholds(records)
    sem_scores = [r["semantic"] for r in rows]
    totals = {
        "p95_latency_ms": p95([r["latency_ms"] for r in rows]),
        "mean_semantic": sum(sem_scores) / len(sem_scores) if sem_scores else 0.0,
        "reply_f1": reply_f1(replies_path) if replies_path and Path(replies_path).exists() else None,
        "safety_violations": sum(r["safety"] for r in rows),
        "hard_mismatches": sum(1 for r in rows for ok in r["hard"].values() if not ok),
    }
    checks = {
        "p95_latency_ms": totals["p95_latency_ms"] <= thresholds["p95_latency_ms"],
        "mean_semantic": totals["mean_semantic"] >= thresholds["personalization_score_min"],
        "reply_f1": totals["reply_f1"] is None or totals["reply_f1"] >= thresholds["reply_classification_f1_min"],
        "safety_violations": totals["safety_violations"] <= thresholds["safety_violations_max"],
        "hard_fields": totals["hard_mismatches"] == 0,
    }
    return {"rows": rows, "thresholds": thresholds, "totals": totals, "checks": checks, "passed": all(checks.values())}


def format_report(report: dict[str, Any]) -> str:
    mark = lambda ok: "ok" if ok else "FAIL"  # noqa: E731
    header = f"{'task':32} {'send':5} {'chan':4} {'at':4} {'cta':4} {'act':4} {'st':4} {'sem':5} {'safe':4} {'try':3} {'ms':>6}"
    lines = [header, "-" * len(header)]
    for r in rows_sorted(report["rows"]):
        h = r["hard"]
        lines.append(
            f"{r['task_id'][:32]:32} {mark(h['send']):5} {mark(h['channel']):4} {mark(h['send_at']):4} "
            f"{mark(h['cta']):4} {mark(h['next_action']):4} {mark(h['states']):4} {r['semantic']:5.2f} "
            f"{r['safety']:4d} {r['attempts']:3d} {r['latency_ms']:6d}"
        )
    t, th, c = report["totals"], report["thresholds"], report["checks"]
    f1 = "n/a" if t["reply_f1"] is None else f"{t['reply_f1']:.2f}"
    lines.append("")
    lines.append(
        f"p95 latency {t['p95_latency_ms']:.0f} ms (max {th['p95_latency_ms']}) {mark(c['p95_latency_ms'])} | "
        f"mean semantic {t['mean_semantic']:.2f} (min {th['personalization_score_min']}) {mark(c['mean_semantic'])} | "
        f"reply macro-F1 {f1} (min {th['reply_classification_f1_min']}) {mark(c['reply_f1'])} | "
        f"safety violations {t['safety_violations']} (max {th['safety_violations_max']}) {mark(c['safety_violations'])} | "
        f"hard-field mismatches {t['hard_mismatches']} {mark(c['hard_fields'])}"
    )
    lines.append("RESULT: " + ("PASS" if report["passed"] else "FAIL"))
    return "\n".join(lines)


def rows_sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows  # input order


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score agent output against expected blocks.")
    parser.add_argument("--in", dest="in_path", required=True, help="JSONL with expected blocks")
    parser.add_argument("--out", dest="out_path", help="existing agent output JSONL; omit to run the agent now")
    parser.add_argument("--replies", default=str(Path(__file__).resolve().parent / "tests" / "replies.jsonl"))
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM semantic judge")
    args = parser.parse_args(argv)
    records = load_jsonl(args.in_path)
    outputs = {o["task_id"]: o for o in load_jsonl(args.out_path)} if args.out_path else None
    report = evaluate(records, outputs, use_judge=not args.no_judge, replies_path=args.replies)
    print(format_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
