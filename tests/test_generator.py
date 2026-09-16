"""Offline tests for llm.py (stub provider, logging) and generator.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import generator
import llm
import validators
from models import Record
from policy import resolve

ROOT = Path(__file__).resolve().parent.parent
RAW = [json.loads(l) for l in (ROOT / "sample.jsonl").read_text().splitlines() if l.strip()]
DECISIONS = {r["task_id"]: resolve(Record.model_validate(r)) for r in RAW}
RECORDS = {r["task_id"]: Record.model_validate(r) for r in RAW}


@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("LLM_STUB_RESPONSE", raising=False)
    return tmp_path


def test_prompt_contains_required_sections():
    d = DECISIONS["prospect_long_horizon_day3"]
    system, user = generator.build_prompt(d)
    p = system + user
    for needle in ["# Role", "# Hard rules", "Fair housing", "PII", "<facts>", "</facts>", "Channel: email",
                   "# Examples", "Reply 1 for Thu, 2 for Fri", "https://oakridge.example/tour", "# Output",
                   '{"subject": string, "body": string}', d.opt_out_line, "Untrusted data"]:
        assert needle in p, needle
    assert "<facts>" not in system and "# Hard rules" not in user  # static/dynamic split
    assert "24/7 fitness" not in p  # invented detail scrubbed from the few-shot example
    assert "# Feedback" not in p
    facts = json.loads(p.split("<facts>")[1].split("</facts>")[0])
    assert facts["channel"] == "email" and facts["cta"] == {"type": "schedule_tour", "link": "https://oakridge.example/tour"}
    assert facts["amenities"] == ["pool", "fitness"] and facts["move_timeframe"] == "mid-February"
    assert "last_interaction" not in p and "expected" not in p  # raw record never reaches the prompt


def test_prompt_is_under_600_words_before_facts():
    d = DECISIONS["prospect_welcome_day0"]
    system, user = generator.build_prompt(d)
    p = system + user
    prose = p.split("<facts>")[0] + p.split("</facts>")[1]
    assert len(prose.split()) < 600, len(prose.split())


def test_prompt_feedback_section():
    d = DECISIONS["prospect_welcome_day0"]
    prev = generator.Draft(subject=None, body="bad")
    system, p = generator.build_prompt(d, feedback=["opt_out_missing_or_not_last", "tone_emoji"], previous=prev)
    assert "Your previous draft failed these checks, fix them" in p and "# Feedback" not in system
    assert "- opt_out_missing_or_not_last" in p and "- tone_emoji" in p and '"body": "bad"' in p


def test_sms_prompt_channel_rules():
    system, user = generator.build_prompt(DECISIONS["prospect_welcome_day0"])
    assert "Channel: sms" in user and "Reply 1 for Thu, 2 for Fri" in user and "hard limit 320" in system
    assert "Learn more" not in user


def test_stub_drafts_pass_validators_for_both_samples():
    for task_id, d in DECISIONS.items():
        draft = generator.write(d)
        assert validators.run(draft, d, RECORDS[task_id]) == [], (task_id, draft)


def test_stub_sms_subject_is_none():
    draft = generator.write(DECISIONS["prospect_welcome_day0"])
    assert draft.subject is None and "\n" not in draft.body


def test_complete_logs_prompt_and_response(stub_env):
    d = DECISIONS["prospect_welcome_day0"]
    generator.write(d)
    log = stub_env / "logs" / "prospect_welcome_day0.jsonl"
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(entries) == 1
    e = entries[0]
    assert e["provider"] == "stub" and e["task_id"] == "prospect_welcome_day0"
    assert "<facts>" in e["prompt"] and e["response"]["body"] and e["error"] is None
    assert "# Hard rules" in e["system"] and e["usage"] is None  # stub reports no token usage
    assert isinstance(e["latency_ms"], int)


def test_stub_override_and_error_logging(monkeypatch, stub_env):
    monkeypatch.setenv("LLM_STUB_RESPONSE", '{"subject": "", "body": "override"}')
    assert llm.complete("x", {}, task_id="t1") == {"subject": "", "body": "override"}
    monkeypatch.setenv("LLM_STUB_RESPONSE", '"not an object"')
    with pytest.raises(llm.LLMError):
        llm.complete("x", {}, task_id="t1")
    entries = [json.loads(l) for l in (stub_env / "logs" / "t1.jsonl").read_text().splitlines()]
    assert entries[1]["error"] and entries[1]["response"] is None


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(llm.LLMError):
        llm.complete("x", {}, task_id="t2")


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.chdir("/")  # no .env here
    s = llm.settings()
    assert s["provider"] == "anthropic" and s["model"] == "claude-opus-5" and s["temperature"] == 0.2
    assert not llm._supports_temperature("claude-opus-5") and llm._supports_temperature("claude-haiku-4-5")
    assert llm._supports_temperature("claude-sonnet-4-5") and llm._supports_temperature("claude-sonnet-4-6")
    assert llm._supports_effort("claude-opus-5") and llm._supports_effort("claude-sonnet-4-6")
    assert not llm._supports_effort("claude-sonnet-4-5") and not llm._supports_effort("claude-haiku-4-5")


def test_load_dotenv_does_not_override(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("LLM_MODEL=from-dotenv\nexport LLM_EFFORT='medium'\n# comment\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "from-env")
    monkeypatch.delenv("LLM_EFFORT", raising=False)
    llm.load_dotenv()
    import os
    assert os.environ["LLM_MODEL"] == "from-env" and os.environ["LLM_EFFORT"] == "medium"


def test_write_rejects_missing_body(monkeypatch):
    monkeypatch.setenv("LLM_STUB_RESPONSE", '{"subject": ""}')
    with pytest.raises(llm.LLMError):
        generator.write(DECISIONS["prospect_welcome_day0"])
