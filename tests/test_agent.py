"""End-to-end tests for agent.py with the stub provider."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import agent
import generator
import llm
from models import Draft

ROOT = Path(__file__).resolve().parent.parent
LINES = [l for l in (ROOT / "sample.jsonl").read_text().splitlines() if l.strip()]
RAW = [json.loads(l) for l in LINES]
REQUIRED = {"consent_verified", "fair_housing_check_passed", "brand_style_applied"}


@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("LLM_STUB_RESPONSE", raising=False)
    return tmp_path


def test_sample_end_to_end_matches_expected_structure():
    outs = list(agent.process_lines(LINES))
    assert [o["task_id"] for o in outs] == [r["task_id"] for r in RAW]
    for out, raw in zip(outs, RAW):
        exp = raw["expected"]
        assert out["decision"]["send"] is True
        assert REQUIRED <= set(out["states_verified"])
        nm = out["next_message"]
        assert nm["channel"] == exp["next_message"]["channel"]
        assert nm["send_at"] == exp["next_message"]["send_at"]
        assert nm["cta"] == exp["next_message"]["cta"]
        assert nm["body"].endswith(exp["next_message"]["body"].split(". ")[-1])  # same opt-out line
        assert (nm["subject"] is None) == (exp["next_message"]["subject"] is None)
        assert out["next_action"] == exp["next_action"]
        v = out["validation"]
        assert v["opt_out_present"] and not v["pii_leak"] and v["fair_housing_flags"] == [] and v["attempts"] == 1
        assert isinstance(out["latency_ms"], int)
    assert outs[0]["validation"]["sms_length"] <= 320 and outs[1]["validation"]["sms_length"] is None
    assert set(outs[0]) == {"task_id", "decision", "states_verified", "next_message", "next_action", "validation", "latency_ms"}


def test_no_send_record_emits_null_message():
    raw = dict(RAW[0], lifecycle_stage="do_not_contact")
    out = agent.process_record(raw)
    assert out["decision"] == {"send": False, "reason": "do_not_contact"}
    assert out["next_message"] is None and out["next_action"] == {"type": "mark_uncontactable"}
    assert out["states_verified"] == [] and out["validation"]["attempts"] == 0


def test_validation_failure_fails_closed_after_three_attempts(monkeypatch):
    monkeypatch.setenv("LLM_STUB_RESPONSE", '{"subject": "", "body": "Perfect for families! Call 972-555-0142 🎉"}')
    out = agent.process_record(RAW[0])
    assert out["decision"] == {"send": False, "reason": "validation_failed"}
    assert out["next_message"] is None and out["next_action"] == {"type": "flag_for_review"}
    assert out["validation"]["attempts"] == 3 and out["validation"]["pii_leak"] is True
    assert out["validation"]["fair_housing_flags"] and not out["validation"]["opt_out_present"]
    assert out["states_verified"] == ["consent_verified"]  # copy states never claimed


def test_generation_failure_fails_closed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    out = agent.process_record(RAW[0])
    assert out["decision"] == {"send": False, "reason": "generation_failed"}
    assert out["next_message"] is None and out["validation"]["attempts"] == 3


def test_retry_feeds_violations_back_and_succeeds(monkeypatch):
    calls = []
    good = generator.write(agent.policy.resolve(agent.Record.model_validate(RAW[0])))

    def fake_write(decision, feedback=None, previous=None):
        calls.append((feedback, previous))
        return Draft(subject=None, body="Hi Taylor, Oak Ridge tours!!") if len(calls) == 1 else good

    monkeypatch.setattr(generator, "write", fake_write)
    out = agent.process_record(RAW[0])
    assert out["decision"]["send"] is True and out["validation"]["attempts"] == 2
    assert calls[0] == (None, None)
    assert calls[1][0] and "opt_out_missing_or_not_last" in calls[1][0] and calls[1][1].body.endswith("!!")


def test_generation_error_then_success(monkeypatch):
    calls = []
    good = generator.write(agent.policy.resolve(agent.Record.model_validate(RAW[1])))

    def flaky(decision, feedback=None, previous=None):
        calls.append(1)
        if len(calls) == 1:
            raise llm.LLMError("boom")
        return good

    monkeypatch.setattr(generator, "write", flaky)
    out = agent.process_record(RAW[1])
    assert out["decision"]["send"] is True and out["validation"]["attempts"] == 2


def test_bad_lines_do_not_stop_the_run():
    lines = [LINES[0], "{not json", '"a string"', json.dumps({"persona": "prospect"}), LINES[1]]
    outs = list(agent.process_lines(lines))
    assert len(outs) == 5
    assert outs[0]["decision"]["send"] and outs[4]["decision"]["send"]
    assert outs[1]["decision"]["reason"] == "invalid_json" and outs[1]["task_id"] == "unknown-line-2"
    assert outs[2]["decision"]["reason"] == "invalid_record"
    assert outs[3]["decision"]["reason"] == "invalid_record"  # task_id is required


def test_unexpected_exception_yields_agent_error(monkeypatch):
    monkeypatch.setattr(agent.policy, "resolve", lambda r: (_ for _ in ()).throw(RuntimeError("kaboom")))
    outs = list(agent.process_lines([LINES[0]]))
    assert outs[0]["decision"] == {"send": False, "reason": "agent_error"} and outs[0]["task_id"] == RAW[0]["task_id"]


def test_run_with_files_and_blank_lines(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(LINES[0] + "\n\n" + LINES[1] + "\n")
    dst = tmp_path / "out" / "o.jsonl"
    assert agent.run(str(src), str(dst)) == 0
    outs = [json.loads(l) for l in dst.read_text().splitlines()]
    assert [o["task_id"] for o in outs] == [r["task_id"] for r in RAW]


def test_cli_stdin_stdout(stub_env):
    env = dict(os.environ, LLM_PROVIDER="stub", LLM_LOG_DIR=str(stub_env / "logs"))
    proc = subprocess.run([sys.executable, str(ROOT / "agent.py")], input="\n".join(LINES), capture_output=True,
                          text=True, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    outs = [json.loads(l) for l in proc.stdout.splitlines()]
    assert len(outs) == 2 and all(o["decision"]["send"] for o in outs)
    assert "2 records, 2 sent" in proc.stderr
