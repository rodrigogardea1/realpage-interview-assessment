"""Rules learned from holdout.jsonl, each tested against the record that evidences it.
The holdout is training data here: policy must reproduce every hard field except the
send-time minutes, which the input does not determine. No rule may key on task_id."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import agent
import evaluate
import generator
import policy
import validators
from models import EMPTY_MESSAGE, Draft, Record
from policy import STAGES, UNKNOWN_STAGE, cta_for, final_states, normalize_unit, resolve, resolve_anchor

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = [json.loads(l) for l in (ROOT / "holdout.jsonl").read_text().splitlines() if l.strip()]
BY_ID = {r["task_id"]: r for r in HOLDOUT}
UNSEEN = [json.loads(l) for l in (ROOT / "tests" / "unseen.jsonl").read_text().splitlines() if l.strip()]
AS_OF = datetime.fromisoformat("2025-12-09T08:00:00-06:00")
SENDS = [r for r in HOLDOUT if r["expected"]["next_message"]["channel"] != "none"]
# send_at matches to the minute only where the reference sits on a channel window.
EXACT_SEND_AT = {"prospect_welcome_day0", "prospect_long_horizon_day3", "resident_renewal_undecided_followup"}


def rec(task_id: str, **overrides) -> Record:
    raw = copy.deepcopy(BY_ID[task_id])
    raw.pop("expected", None)
    for key, value in overrides.items():
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if value is DELETE:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
    return Record.model_validate(raw)


DELETE = object()


@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("LLM_STUB_RESPONSE", raising=False)
    monkeypatch.delenv("AGENT_AS_OF", raising=False)


# ------------------------------------------------------ every record, hard fields


def test_holdout_has_twelve_records_with_expected_blocks():
    assert len(HOLDOUT) == 12 and all("expected" in r for r in HOLDOUT)


@pytest.mark.parametrize("raw", HOLDOUT, ids=[r["task_id"] for r in HOLDOUT])
def test_policy_reproduces_every_hard_field_except_minutes(raw):
    exp, msg = raw["expected"], raw["expected"]["next_message"]
    d = resolve(Record.model_validate(raw), as_of=AS_OF)
    required = set(raw["assertions"]["required_states"])
    assert d.next_action == exp["next_action"]
    if msg["channel"] == "none":
        assert d.send is False and d.empty_message and required <= set(d.states_verified)
        return
    assert d.send is True
    assert d.channel == msg["channel"]
    assert d.cta.to_output() == msg["cta"]
    assert d.send_at.isoformat()[:10] == msg["send_at"][:10]              # same local day, always
    assert (d.send_at.isoformat() == msg["send_at"]) == (raw["task_id"] in EXACT_SEND_AT)
    assert d.send_at.strftime("%H:%M") == {"sms": "09:00", "email": "10:00"}[d.channel]   # windows kept
    assert required <= set(final_states(d, validated=True))


def test_no_rule_keys_on_task_id():
    for raw in HOLDOUT:
        renamed = dict(copy.deepcopy(raw), task_id="zzz")
        a = resolve(Record.model_validate(raw), as_of=AS_OF).model_dump(exclude={"task_id"})
        b = resolve(Record.model_validate(renamed), as_of=AS_OF).model_dump(exclude={"task_id"})
        assert a == b
    for module in ("policy.py", "validators.py", "classifier.py"):
        source = (ROOT / module).read_text()
        lines = [l for l in source.splitlines() if "task_id" in l]
        assert all("task_id=record.task_id" in l for l in lines), (module, lines)   # identifier pass-through only


# ------------------------------------------------------------- 1. anchor


def test_anchor_priority_last_interaction_then_as_of_then_input_then_now():
    tz = policy.ZoneInfo("America/Chicago")
    now = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)
    with_li = Record.model_validate(BY_ID["prospect_welcome_day0"]).input
    assert resolve_anchor(with_li, tz, AS_OF, now)[1] == "last_interaction"
    no_show = rec("prospect_no_show_reengage").input
    assert resolve_anchor(no_show, tz, AS_OF, now) == (AS_OF, "as_of")
    anchor, source = resolve_anchor(no_show, tz, None, now)
    assert source == "input.missed_tour_time" and anchor.isoformat() == "2025-12-08T14:00:00-06:00"
    bare = rec("resident_loyalty_engage").input
    assert resolve_anchor(bare, tz, None, now) == (now.astimezone(tz), "now")
    assert resolve_anchor(rec("resident_welcome_day0").input, tz, None, now)[1] == "now"   # date-only fields are not anchors


def test_no_show_lands_on_the_right_day_even_without_as_of():
    d = resolve(rec("prospect_no_show_reengage"))
    assert d.send and d.send_at.isoformat() == "2025-12-09T09:00:00-06:00"
    assert d.anchor_source == "input.missed_tour_time"
    assert "anchor 2025-12-08T14:00:00-06:00 (input.missed_tour_time)" in d.reason


def test_missing_anchor_never_blocks_and_reason_names_the_source():
    d = resolve(rec("resident_loyalty_engage"), now=datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc))
    assert d.send and d.anchor_source == "now" and "(now)" in d.reason
    assert d.send_at.isoformat() == "2026-01-09T10:00:00-06:00"          # noon + 3 days is past 10:00 -> next day
    d = resolve(rec("resident_loyalty_engage"), as_of=AS_OF)
    assert "anchor 2025-12-09T08:00:00-06:00 (as_of)" in d.reason
    sample = resolve(Record.model_validate(BY_ID["prospect_welcome_day0"]), as_of=AS_OF)
    assert "anchor" not in sample.reason and sample.anchor_source == "last_interaction"   # as_of never overrides it


def test_as_of_flag_and_env(tmp_path):
    assert agent.parse_as_of(None) is None and agent.parse_as_of(" ") is None
    assert agent.parse_as_of("2025-12-09T14:00:00Z") == datetime(2025, 12, 9, 14, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        agent.parse_as_of("tomorrow")
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(BY_ID["resident_loyalty_engage"]))
    env = dict(os.environ, LLM_PROVIDER="stub", LLM_LOG_DIR=str(tmp_path / "logs"))
    flag = subprocess.run([sys.executable, str(ROOT / "agent.py"), "--in", str(src), "--as-of", "2025-12-09T08:00:00-06:00"],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    assert json.loads(flag.stdout)["next_message"]["send_at"] == "2025-12-12T10:00:00-06:00"
    via_env = subprocess.run([sys.executable, str(ROOT / "agent.py"), "--in", str(src)], capture_output=True, text=True,
                             env=dict(env, AGENT_AS_OF="2025-12-20T08:00:00-06:00"), cwd=ROOT)
    assert json.loads(via_env.stdout)["next_message"]["send_at"] == "2025-12-23T10:00:00-06:00"
    bad = subprocess.run([sys.executable, str(ROOT / "agent.py"), "--in", str(src), "--as-of", "nope"],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    assert bad.returncode == 2 and "ISO-8601" in bad.stderr


# -------------------------------------------------------- 2. stage table


def test_every_holdout_stage_is_a_table_row():
    for raw in HOLDOUT:
        assert (raw["persona"], raw["lifecycle_stage"]) in STAGES, raw["task_id"]
    for entry in list(STAGES.values()) + [UNKNOWN_STAGE]:
        assert {"delay_days", "primary_cta", "next_action", "intent"} <= set(entry)


def test_prospect_open_gap_applies_only_after_a_last_interaction():
    r2 = resolve(Record.model_validate(BY_ID["prospect_long_horizon_day3"]), as_of=AS_OF)
    assert r2.send_at.isoformat() == "2025-12-09T10:00:00-06:00"         # 12/06 + 3 days
    es = resolve(rec("prospect_spanish_locale"), as_of=AS_OF)
    assert es.send_at.isoformat() == "2025-12-09T09:00:00-06:00"         # no last touch -> due now
    assert es.horizon == "unknown" and es.next_action == {"type": "follow_up_in_days", "value": 2}


def test_new_prospect_without_move_date_starts_the_long_cadence():
    d = resolve(rec("prospect_consent_block_sms_fallback_email"), as_of=AS_OF)
    assert d.next_action == {"type": "start_cadence", "name": "prospect_welcome_long_horizon"}
    assert d.cta.to_output() == {"type": "schedule_tour", "link": "https://oakridge.example/tour"}   # default primary_cta
    assert d.reason.startswith("email consented (sms preferred, not consented); prospect new; horizon unknown")


def test_resident_welcome_sends_two_days_before_move_in():
    assert resolve(rec("resident_welcome_day0"), as_of=AS_OF).send_at.isoformat() == "2025-12-10T10:00:00-06:00"
    no_date = resolve(rec("resident_welcome_day0", **{"input.move_in_date": DELETE}), as_of=AS_OF)
    assert no_date.send_at.isoformat() == "2025-12-10T10:00:00-06:00"    # else +1 day
    late = resolve(rec("resident_welcome_day0", **{"input.move_in_date": "2025-12-10"}), as_of=AS_OF)
    assert late.send_at.isoformat() == "2025-12-09T10:00:00-06:00"       # two days before has passed -> next window


def test_stage_delays():
    at = lambda tid: resolve(rec(tid), as_of=AS_OF).send_at.isoformat()   # noqa: E731
    assert at("resident_renewal_90day_notice") == "2025-12-09T10:00:00-06:00"          # +0
    assert at("resident_renewal_undecided_followup") == "2025-12-14T09:00:00-06:00"    # +5
    assert at("resident_renewal_details_branch_email") == "2025-12-14T10:00:00-06:00"  # +5
    assert at("resident_loyalty_engage") == "2025-12-12T10:00:00-06:00"                # +3
    assert at("prospect_cancellation_manager_cross_sell") == "2025-12-09T10:00:00-06:00"


def test_unknown_stage_defaults():
    d = resolve(rec("resident_loyalty_engage", lifecycle_stage="something_new",
                    **{"assertions.constraints.primary_cta": "schedule_callback"}), as_of=AS_OF)
    assert d.send and d.send_at.isoformat() == "2025-12-09T10:00:00-06:00"
    assert d.next_action == {"type": "follow_up_in_days", "value": 3} and d.message_intent is None
    assert d.cta.to_output() == {"type": "schedule_callback", "link": "https://oakridge.example/schedule-callback"}


def test_next_action_templates_are_not_shared_between_calls():
    a = resolve(rec("resident_renewal_undecided_followup"), as_of=AS_OF).next_action
    a["mapping"]["yes"] = "tampered"
    assert resolve(rec("resident_renewal_undecided_followup"), as_of=AS_OF).next_action["mapping"]["yes"] == "start_esign_flow"


# ---------------------------------------------------------- 3. CTA catalog


def test_tour_options_follow_the_record_language():
    assert resolve(rec("prospect_spanish_locale"), as_of=AS_OF).cta.options == ["jueves", "viernes"]
    weekend = resolve(rec("prospect_spanish_locale", **{"input.tour_availability": ["Sat", "Sun"]}), as_of=AS_OF)
    assert weekend.cta.options == ["sábado", "domingo"] and weekend.tour_days == ["Sat", "Sun"]
    assert resolve(Record.model_validate(BY_ID["prospect_welcome_day0"])).cta.options == ["Thu", "Fri"]


def test_unit_links_use_a_regular_hyphen():
    assert BY_ID["resident_renewal_90day_notice"]["input"]["unit"] == "A‑204"      # U+2011 in the data
    assert normalize_unit("A‑204") == "A-204" and normalize_unit(" b–118 ") == "b-118" and normalize_unit(None) is None
    d = resolve(rec("resident_renewal_90day_notice"), as_of=AS_OF)
    assert d.unit == "A-204" and d.cta.link == "https://oakridge.example/renewal/A-204"
    details = resolve(rec("resident_renewal_details_branch_email"), as_of=AS_OF)
    assert details.cta.link == "https://oakridge.example/renewal/A-204/details"
    no_unit = resolve(rec("resident_renewal_details_branch_email", **{"input.unit": DELETE}), as_of=AS_OF)
    assert no_unit.cta.link == "https://oakridge.example/renewal/details"


def test_cta_shapes_by_channel():
    assert cta_for("reschedule", "sms", "Oak Ridge Apartments").to_output() == {"type": "reschedule", "options": ["today", "tomorrow"]}
    assert cta_for("reschedule", "email", "Oak Ridge Apartments").to_output() == {"type": "reschedule", "link": "https://oakridge.example/tour"}
    assert cta_for("intent_capture", "email", "Oak Ridge Apartments").to_output() == {"type": "intent_capture", "options": ["yes", "no", "details"]}
    assert cta_for("reschedule", "sms", "X", language="es").options == ["hoy", "mañana"]
    assert policy.cta_type_for("reschedule_tour", "prospect") == "reschedule"
    assert policy.cta_type_for("reply_intent", "resident") == "intent_capture"


# ------------------------------------------------------ 4. no-consent shape


def test_no_consent_line_matches_the_data_exactly():
    raw = BY_ID["resident_opt_out_respected"]
    out = agent.process_record(raw, AS_OF)
    assert out["next_message"] == raw["expected"]["next_message"] == EMPTY_MESSAGE
    assert out["next_action"] == raw["expected"]["next_action"] == {"type": "no_op", "reason": "no_contact_consent"}
    assert out["decision"] == {"send": False, "reason": "no_consented_channel"}
    assert out["states_verified"] == ["consent_verified"] and out["validation"]["attempts"] == 0
    hard = evaluate.hard_fields(out, raw["expected"], raw["assertions"]["required_states"])
    assert all(hard.values()), hard
    assert evaluate.judge_body(out, raw["expected"], "t", use_judge=False) == 1.0
    null_line = dict(out, next_message=None)
    assert not evaluate.hard_fields(null_line, raw["expected"], ["consent_verified"])["channel"]   # null is not the shape


def test_other_no_sends_keep_a_null_message():
    out = agent.process_record(dict(BY_ID["prospect_welcome_day0"], lifecycle_stage="do_not_contact"))
    assert out["next_message"] is None


# ------------------------------------------------------- 5. required states


def test_renewal_offer_loaded():
    for tid in ("resident_renewal_90day_notice", "resident_renewal_undecided_followup"):
        assert "renewal_offer_loaded" in resolve(rec(tid), as_of=AS_OF).states_verified   # lease_end_date / renewal_offer_id
    assert "renewal_offer_loaded" not in resolve(rec("resident_loyalty_engage"), as_of=AS_OF).states_verified
    d = resolve(rec("resident_renewal_undecided_followup", **{"input.renewal_offer_id": DELETE}), as_of=AS_OF)
    assert not d.send and d.reason == "unverifiable_required_state:renewal_offer_loaded"
    assert d.next_action == {"type": "flag_for_review"}


def test_unknown_required_state_blocks_the_send():
    d = resolve(rec("prospect_no_show_reengage", **{"assertions.required_states": ["consent_verified", "credit_check_passed"]}), as_of=AS_OF)
    assert not d.send and d.reason == "unverifiable_required_state:credit_check_passed"
    assert d.states_verified == ["consent_verified"]


def test_locale_applied_needs_validated_copy():
    d = resolve(rec("prospect_spanish_locale"), as_of=AS_OF)
    assert "locale_applied" not in final_states(d, validated=False)
    assert "locale_applied" in final_states(d, validated=True)
    assert "locale_applied" not in final_states(resolve(Record.model_validate(BY_ID["prospect_welcome_day0"])), True)
    out = agent.process_record(BY_ID["prospect_spanish_locale"], AS_OF)
    assert out["decision"]["send"] and "locale_applied" in out["states_verified"]


# ------------------------------------------------------------ 6. validators

KNOWN_BODY_EXCEPTIONS = {   # these two name the unit but never the property; check_personalization is unchanged
    "resident_renewal_undecided_followup": ["personalization_missing_property"],
    "resident_renewal_details_branch_email": ["personalization_missing_property"],
}


@pytest.mark.parametrize("raw", SENDS, ids=[r["task_id"] for r in SENDS])
def test_holdout_expected_bodies_against_validators(raw):
    record = Record.model_validate(raw)
    msg = raw["expected"]["next_message"]
    found = validators.run(Draft(subject=msg["subject"], body=msg["body"]), resolve(record, as_of=AS_OF), record)
    assert found == KNOWN_BODY_EXCEPTIONS.get(raw["task_id"], [])


def test_resident_own_unit_is_allowed_but_nobody_elses():
    resident = rec("resident_renewal_90day_notice", **{"input.unit": "204"})
    assert validators.check_pii("Renewal options for Unit 204 are ready", resident) is None
    assert validators.check_pii("Renewal options for Unit 310 are ready", resident) == "pii_unit"
    prospect = rec("prospect_no_show_reengage", **{"input.unit": "204"})
    assert validators.check_pii("Your Unit 204 is ready", prospect) == "pii_unit"            # prospects have no unit
    own = rec("resident_renewal_90day_notice")
    assert validators.check_pii("your home (A‑204) and A-204 again", own) is None             # both hyphen forms


def test_accepted_opt_out_variants():
    ok = validators.check_opt_out
    for line in ["To opt out of emails, click here or reply STOP.", "Opt-out here or reply STOP.",
                 "Opt‑out here or reply STOP.", "Opt-out any time.", "Opt‑out any time."]:
        assert ok(f"Hi Jordan,\\nBody.\\n{line}", "To opt out of emails, click here or reply STOP.", "email", "en") is None, line
    assert ok("Body. Opt-out any time.", "Reply STOP to opt out.", "sms", "en") == "opt_out_missing_or_not_last"
    assert ok("Hola. Reply STOP to opt out.", "Responde STOP para cancelar.", "sms", "es") == "opt_out_missing_or_not_last"
    assert ok("Hola. Responde STOP para cancelar.", "Responde STOP para cancelar.", "sms", "es") is None
    assert ok("Opt-out any time. One more thing.", "x", "email", "en") == "opt_out_missing_or_not_last"


def test_numbered_replies_required_for_every_options_cta():
    d = resolve(rec("prospect_no_show_reengage"), as_of=AS_OF)
    assert validators.check_cta(Draft(body="Reschedule? Reply 1 for today, 2 for tomorrow."), d) is None
    assert validators.check_cta(Draft(body="Reschedule today or tomorrow?"), d) == "cta_reply_number_missing:1"
    intent = resolve(rec("resident_renewal_undecided_followup"), as_of=AS_OF)
    assert validators.check_cta(Draft(body="Renew? Reply 1 Yes, 2 No, 3 Need more details."), intent) is None
    assert validators.check_cta(Draft(body="Renew? Reply 1 Yes, 2 No."), intent) == "cta_option_missing:details"


def test_spanish_copy_must_read_as_spanish():
    es_body = BY_ID["prospect_spanish_locale"]["expected"]["next_message"]["body"]
    assert validators.check_language(es_body, "es") is None
    assert validators.check_language(BY_ID["prospect_welcome_day0"]["expected"]["next_message"]["body"], "es") == "locale_mismatch:es"
    assert validators.check_language("anything at all", "en") is None


# ---------------------------------------------------------- 7. generator


def test_message_intent_and_unit_reach_the_facts():
    system, user = generator.build_prompt(resolve(rec("resident_renewal_90day_notice"), as_of=AS_OF))
    facts = json.loads(user.split("<facts>")[1].split("</facts>")[0])
    assert facts["unit"] == "A-204"
    assert facts["message_intent"] == "Lease ends soon; invite them to review renewal options for their unit."
    assert facts["extra_context"] == {"tenure_months": "10", "lease_end_date": "2026-03-10"}
    assert "REN" not in user                                           # the offer id never reaches the prompt
    missed = generator.build_prompt(resolve(rec("prospect_no_show_reengage"), as_of=AS_OF))[1]
    assert "missed a scheduled tour" in missed and 'Reply 1 for today, 2 for tomorrow.' in missed


def test_prompt_carries_the_holdout_examples_and_the_invention_warning():
    system, _ = generator.build_prompt(resolve(rec("prospect_no_show_reengage"), as_of=AS_OF))
    assert "‑" not in system                                      # regular hyphens only
    for needle in ["We missed you yesterday, Taylor.", "Responde 1 para jueves, 2 para viernes.",
                   "Reply 1 Yes, 2 No, 3 Need more details.", "https://oakridge.example/renewal/A-204/details",
                   "https://oakridge.example/welcome", "https://oakridge.example/loyalty", "Book your time here:",
                   "SMS, prospect:", "SMS, resident:", "Email, prospect:", "Email, resident:"]:
        assert needle in system, needle
    assert '"$1,650" in the cancelled-tour email and "10 days" in the renewal email' in system
    assert "must never be imitated" in system and "Never invent pricing, offers" in system
    assert "10. Purpose: `message_intent`" in system


def test_spanish_cta_instruction():
    text = generator.cta_instruction(resolve(rec("prospect_spanish_locale"), as_of=AS_OF))
    assert "Responde 1 para jueves, 2 para viernes." in text
    loyalty = generator.cta_instruction(resolve(rec("resident_loyalty_engage"), as_of=AS_OF))
    assert '"Enroll → https://oakridge.example/loyalty"' in loyalty


# -------------------------------------------------------------- 8. models


def test_thresholds_accepted_at_either_level_and_new_constraints():
    top = Record.model_validate(BY_ID["resident_renewal_90day_notice"])
    assert top.thresholds["p95_latency_ms"] == 2500 and top.effective_thresholds()["personalization_score_min"] == 0.85
    raw = copy.deepcopy(BY_ID["resident_renewal_90day_notice"])
    raw["assertions"]["thresholds"] = raw.pop("thresholds")
    nested = Record.model_validate(raw)
    assert nested.thresholds == {} and nested.effective_thresholds()["p95_latency_ms"] == 2500
    assert evaluate.strictest_thresholds([raw])["p95_latency_ms"] == 2500
    es = Record.model_validate(BY_ID["prospect_spanish_locale"])
    assert es.assertions.constraints.locale_applied is True and es.thresholds["locale_accuracy_min"] == 0.95
    blocked = Record.model_validate(BY_ID["resident_opt_out_respected"])
    assert blocked.assertions.constraints.respect_consent is True
    assert "respect_consent" not in policy.build_extra_context(blocked) and "locale_applied" not in policy.build_extra_context(es)


# ----------------------------------------------------- end to end on the stub


def test_holdout_runs_end_to_end_on_the_stub():
    outs = list(agent.process_lines((json.dumps(r) for r in HOLDOUT), AS_OF))
    assert [o["task_id"] for o in outs] == [r["task_id"] for r in HOLDOUT]
    for out, raw in zip(outs, HOLDOUT):
        sends = raw["expected"]["next_message"]["channel"] != "none"
        assert out["decision"]["send"] is sends, (out["task_id"], out["decision"])
        if sends:
            assert out["validation"]["attempts"] == 1
            hard = evaluate.hard_fields(out, raw["expected"], raw["assertions"]["required_states"])
            assert [k for k, ok in hard.items() if not ok and k != "send_at"] == [], (out["task_id"], hard)


# ------------------------------------------------ 9. unseen combinations


def test_unseen_file_has_five_records_without_expected_blocks():
    assert len(UNSEEN) == 5 and not any("expected" in r for r in UNSEEN)
    seen = {(r["persona"], r["lifecycle_stage"], r["input"]["language"], tuple(r["channel_preferences"]),
             r["consent"]["sms_opt_in"]) for r in HOLDOUT}
    for r in UNSEEN:
        combo = (r["persona"], r["lifecycle_stage"], r["input"]["language"], tuple(r["channel_preferences"]), r["consent"]["sms_opt_in"])
        assert combo not in seen or "tour_availability" in r["input"], r["task_id"]


@pytest.mark.parametrize("raw", UNSEEN, ids=[r["task_id"] for r in UNSEEN])
def test_unseen_combination_sends_a_validated_message(raw):
    out = agent.process_record(raw, AS_OF)
    assert out["decision"]["send"] is True, out["decision"]
    assert out["validation"]["attempts"] == 1 and out["next_message"]["body"]
    assert set(raw["assertions"]["required_states"]) <= set(out["states_verified"])


def test_unseen_policy_details():
    by = {r["task_id"]: resolve(Record.model_validate(r), as_of=AS_OF) for r in UNSEEN}
    es_renewal = by["unseen_es_resident_renewal_window"]
    assert es_renewal.channel == "sms" and es_renewal.cta.link == "https://oakridge.example/renewal/C-310"
    assert es_renewal.opt_out_line == "Responde STOP para cancelar." and "locale_applied" in es_renewal.post_validation_states
    no_show = by["unseen_no_show_email_only"]
    assert no_show.channel == "email" and no_show.cta.to_output() == {"type": "reschedule", "link": "https://oakridge.example/tour"}
    assert no_show.next_action == {"type": "reset_cadence", "name": "prospect_reengage"}
    assert by["unseen_loyalty_es"].send_at.isoformat() == "2025-12-12T10:00:00-06:00"
    weekend = by["unseen_new_prospect_weekend_tours"]
    assert weekend.cta.options == ["Sat", "Sun"] and weekend.tour_week_phrase == "this week"
    undecided = by["unseen_renewal_undecided_email_fallback"]
    assert undecided.channel == "email" and undecided.cta.to_output() == {"type": "intent_capture", "options": ["yes", "no", "details"]}
    assert undecided.reason.startswith("email consented (sms preferred, not consented); resident renewal_undecided")
