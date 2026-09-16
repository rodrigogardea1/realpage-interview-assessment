"""Every record in tests/edge_cases.jsonl resolves to its expected policy fields,
and the whole file runs end to end on the stub provider without a failed line."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent
from models import Record
from policy import resolve

CASES = [json.loads(l) for l in (Path(__file__).parent / "edge_cases.jsonl").read_text().splitlines() if l.strip()]


@pytest.mark.parametrize("raw", CASES, ids=[c["task_id"] for c in CASES])
def test_policy_matches_expected(raw):
    exp = raw["expected"]
    d = resolve(Record.model_validate(raw))
    assert d.send == exp["decision"]["send"]
    assert d.next_action == exp["next_action"]
    if not d.send:
        assert d.reason == exp["decision"]["reason"]
        return
    msg = exp["next_message"]
    assert d.channel == msg["channel"]
    assert d.send_at.isoformat() == msg["send_at"]
    assert d.cta.to_output() == msg["cta"]


def test_edge_cases_run_end_to_end_on_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "logs"))
    lines = (Path(__file__).parent / "edge_cases.jsonl").read_text().splitlines()
    outs = list(agent.process_lines(lines))
    assert [o["task_id"] for o in outs] == [c["task_id"] for c in CASES]
    for out, raw in zip(outs, CASES):
        assert out["decision"]["send"] == raw["expected"]["decision"]["send"], (out["task_id"], out["decision"])
        assert out["decision"]["reason"] not in ("validation_failed", "generation_failed", "agent_error")
    by_id = {o["task_id"]: o for o in outs}
    assert by_id["edge_injection_first_name"]["next_message"]["body"].startswith("Hi there")
    pii_body = by_id["edge_pii_in_profile"]["next_message"]["body"]
    for leak in ("example.com", "555", "204", "Dell", "80000", "Richardson"):
        assert leak not in pii_body
