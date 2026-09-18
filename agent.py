"""CLI and orchestration: record -> policy.resolve -> generator.write ->
validators.run (retry <= 2 with feedback) -> one output line.

    python agent.py --in sample.jsonl --out out/sample_out.jsonl
    cat sample.jsonl | python agent.py > out.jsonl

One bad record never stops the run: it gets a failed line and the run
continues. Nothing unvalidated is ever emitted (fail closed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, TextIO

from pydantic import ValidationError

import generator
import llm
import policy
import validators
from models import Decision, Draft, NextMessage, OutputLine, Record

MAX_ATTEMPTS = 3  # one draft plus two retries with violations fed back
FAIL_ACTION = {"type": "flag_for_review"}


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _line(decision: Decision, *, draft: Draft | None, violations: list[str], attempts: int,
          validated: bool, reason: str, start: float) -> dict[str, Any]:
    next_message = None
    if validated and draft is not None and decision.cta is not None and decision.send_at is not None:
        next_message = NextMessage(
            channel=decision.channel or "",
            send_at=decision.send_at.isoformat(),
            subject=draft.subject,
            body=draft.body,
            cta=decision.cta.to_output(),
        )
    out = OutputLine(
        task_id=decision.task_id,
        decision={"send": validated, "reason": reason},
        states_verified=policy.final_states(decision, validated),
        next_message=next_message,
        next_action=decision.next_action if validated or not decision.send else FAIL_ACTION,
        validation=validators.summarize(draft, decision, violations, attempts),
        latency_ms=_elapsed_ms(start),
    )
    return out.model_dump()


def _failed_line(task_id: str, reason: str, start: float) -> dict[str, Any]:
    decision = Decision(send=False, reason=reason, next_action=FAIL_ACTION, task_id=task_id)
    return _line(decision, draft=None, violations=[], attempts=0, validated=False, reason=reason, start=start)


def process_record(raw: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    task_id = str(raw.get("task_id") or "unknown") if isinstance(raw, dict) else "unknown"
    try:
        record = Record.model_validate(raw)
    except ValidationError:
        return _failed_line(task_id, "invalid_record", start)

    decision = policy.resolve(record)
    if not decision.send:
        return _line(decision, draft=None, violations=[], attempts=0, validated=False,
                     reason=decision.reason, start=start)

    draft: Draft | None = None
    violations: list[str] = []
    generation_error: str | None = None
    attempts = 0
    for attempts in range(1, MAX_ATTEMPTS + 1):
        try:
            draft = generator.write(decision, feedback=violations or None, previous=draft)
            generation_error = None
        except llm.LLMError as e:
            generation_error = str(e)
            continue
        violations = validators.run(draft, decision, record)
        if not violations:
            return _line(decision, draft=draft, violations=[], attempts=attempts, validated=True,
                         reason=decision.reason, start=start)

    reason = "generation_failed" if generation_error is not None else "validation_failed"
    return _line(decision, draft=draft, violations=violations, attempts=attempts, validated=False,
                 reason=reason, start=start)


def process_lines(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        start = time.perf_counter()
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            yield _failed_line(f"unknown-line-{n}", "invalid_json", start)
            continue
        if not isinstance(raw, dict):
            yield _failed_line(f"unknown-line-{n}", "invalid_record", start)
            continue
        try:
            yield process_record(raw)
        except Exception as e:  # noqa: BLE001 - never let one record kill the run
            print(f"[agent] line {n}: {type(e).__name__}: {e}", file=sys.stderr)
            yield _failed_line(str(raw.get("task_id") or f"unknown-line-{n}"), "agent_error", start)


def split_input(text: str) -> list[str]:
    """JSONL by default; a JSON array (text starts with '[') becomes one line per element.
    Output is always JSONL either way."""
    if text.lstrip().startswith("["):
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return text.splitlines()  # let process_lines report invalid_json per line
        if isinstance(items, list):
            return [json.dumps(item, ensure_ascii=False) for item in items]
    return text.splitlines()


def run(in_path: str | None, out_path: str | None, *, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    source = split_input(Path(in_path).read_text(encoding="utf-8") if in_path else stdin.read())
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sink = Path(out_path).open("w", encoding="utf-8")
    else:
        sink = stdout
    # Warm the real provider's connection once, outside any record's latency
    # clock. Skipped for the stub provider and for empty input.
    if source and llm.settings()["provider"] != "stub":
        llm.warm()
    sent = total = 0
    try:
        for out in process_lines(source):
            sink.write(json.dumps(out, ensure_ascii=False) + "\n")
            sink.flush()
            total += 1
            sent += int(out["decision"]["send"])
    finally:
        if out_path:
            sink.close()
    print(f"[agent] {total} records, {sent} sent, {total - sent} not sent", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Outreach agent: JSONL in, JSONL out.")
    parser.add_argument("--in", dest="in_path", help="input JSONL (default: stdin)")
    parser.add_argument("--out", dest="out_path", help="output JSONL (default: stdout)")
    args = parser.parse_args(argv)
    return run(args.in_path, args.out_path)


if __name__ == "__main__":
    sys.exit(main())
