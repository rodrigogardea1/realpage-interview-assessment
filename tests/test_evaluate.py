"""Offline tests for evaluate.py (judge disabled)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluate

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = evaluate.load_jsonl(ROOT / "sample.jsonl")
REPLIES = ROOT / "tests" / "replies.jsonl"


@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("LLM_STUB_RESPONSE", raising=False)


def test_sample_passes_without_judge():
    report = evaluate.evaluate(SAMPLE, None, use_judge=False, replies_path=REPLIES)
    assert report["passed"], evaluate.format_report(report)
    assert report["totals"]["hard_mismatches"] == 0 and report["totals"]["safety_violations"] == 0
    assert report["totals"]["reply_f1"] == 1.0
    assert report["thresholds"]["personalization_score_min"] == 0.85  # strictest across records
    assert "RESULT: PASS" in evaluate.format_report(report)


def test_hard_field_mismatch_fails_report():
    out = {"decision": {"send": True, "reason": "x"}, "states_verified": ["consent_verified"],
           "next_message": {"channel": "email", "send_at": "2025-12-09T09:00:00-06:00", "subject": None, "body": "b",
                            "cta": {"type": "schedule_tour", "options": ["Thu", "Fri"]}},
           "next_action": {"type": "start_cadence", "name": "prospect_welcome_short_horizon"}}
    hard = evaluate.hard_fields(out, SAMPLE[0]["expected"], SAMPLE[0]["assertions"]["required_states"])
    assert hard == {"send": True, "next_action": True, "states": False, "channel": False, "send_at": True, "cta": True}


def test_no_send_vs_expected_message_scores_zero():
    out = {"decision": {"send": False, "reason": "x"}, "states_verified": [], "next_message": None, "next_action": {}}
    assert evaluate.judge_body(out, SAMPLE[0]["expected"], "t", use_judge=False) == 0.0
    both_null = {"decision": {"send": False, "reason": "x"}, "next_message": None}
    assert evaluate.judge_body(both_null, {"next_message": None}, "t", use_judge=False) == 1.0


def test_macro_f1_and_p95():
    assert evaluate.macro_f1(["a", "b", "a"], ["a", "b", "a"], ("a", "b", "c")) == 1.0
    assert evaluate.macro_f1(["a", "b"], ["b", "a"], ("a", "b")) == 0.0
    assert 0 < evaluate.macro_f1(["a", "a", "b"], ["a", "b", "b"], ("a", "b")) < 1
    assert evaluate.p95([100, 200, 300, 4000]) == 4000
    assert evaluate.p95([]) == 0.0


def test_threshold_failure_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_STUB_RESPONSE", '{"subject": "", "body": "Perfect for families! Call 972-555-0142"}')
    rc = evaluate.main(["--in", str(ROOT / "sample.jsonl"), "--no-judge", "--replies", str(REPLIES)])
    assert rc == 1


def test_scores_existing_output_file(tmp_path):
    import agent
    outs = list(agent.process_lines((ROOT / "sample.jsonl").read_text().splitlines()))
    out_path = tmp_path / "o.jsonl"
    out_path.write_text("\n".join(json.dumps(o) for o in outs))
    rc = evaluate.main(["--in", str(ROOT / "sample.jsonl"), "--out", str(out_path), "--no-judge", "--replies", str(REPLIES)])
    assert rc == 0
