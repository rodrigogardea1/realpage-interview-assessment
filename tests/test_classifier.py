import json
from pathlib import Path

import pytest

from classifier import INTENTS, classify

FIXTURE = [json.loads(l) for l in (Path(__file__).parent / "replies.jsonl").read_text().splitlines() if l.strip()]


@pytest.mark.parametrize("case", FIXTURE, ids=[c["text"][:25] or "<empty>" for c in FIXTURE])
def test_fixture_reply(case):
    assert classify(case["text"]) == case["intent"]


def test_fixture_covers_every_intent():
    assert {c["intent"] for c in FIXTURE} == set(INTENTS)
    assert len(FIXTURE) >= 30


def test_stop_by_is_not_opt_out():
    assert classify("Can I stop by Friday?") == "book_fri"
    assert classify("stop by tomorrow?") == "question"


def test_policy_delegates_to_classifier():
    import policy
    assert policy.classify_reply("  ") is None
    assert policy.classify_reply("1 please") == "book_thu"
    assert policy.classify_reply("unsubscribe") == "stop"
