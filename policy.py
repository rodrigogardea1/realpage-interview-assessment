"""Deterministic decision layer. No LLM calls, no I/O.

Every rule here is cited in the CLAUDE.md rules table. `resolve()` is the
single entry point: it turns a Record into a Decision that says whether to
send, on which channel, when, with which CTA and next action, plus the
sanitized facts the generator may use for copy.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from classifier import Intent as ReplyIntent, classify
from models import Channel, Consent, Cta, Decision, Horizon, Record

# ------------------------------------------------------------------ constants

WINDOW_HOUR: dict[str, int] = {"sms": 9, "email": 10}          # R1, R2
SENDABLE_CHANNELS: tuple[str, ...] = ("sms", "email")
DEFAULT_CHANNEL_ORDER: tuple[str, ...] = ("sms", "email")      # assumption: empty prefs
HORIZON_SHORT_MAX_DAYS = 60                                      # assumption: 32 < 60 < 68
FOLLOW_UP_DAYS: dict[str, int] = {"short": 2, "long": 3, "unknown": 3}
RESIDENT_FOLLOW_UP_DAYS = 3
TOUR_DAYS: list[str] = ["Thu", "Fri"]                            # R1
CTA_TYPE_MAP: dict[str, str] = {"book_tour": "schedule_tour"}    # R1, R2
CTA_LINK_PATH: dict[str, str] = {
    "schedule_tour": "tour",                                     # R2
    "renew_lease": "renew",
    "schedule_maintenance": "maintenance",
    "pay_balance": "pay",
}
DEFAULT_CTA: dict[str, str] = {"prospect": "schedule_tour", "resident": "contact_office"}
SMS_MAX_CHARS = 320
POLICY_STATES: list[str] = ["consent_verified"]
POST_VALIDATION_STATES: list[str] = ["fair_housing_check_passed", "brand_style_applied"]
PROPERTY_SUFFIXES: frozenset[str] = frozenset(
    {"apartments", "apartment", "apts", "residences", "homes", "community", "flats", "lofts", "towers"}
)
OPT_OUT_LINES: dict[str, dict[str, str]] = {
    "en": {
        "sms": "Reply STOP to opt out.",                                        # R1
        "email": "To opt out of emails, click here or reply STOP.",           # R2
    },
    "es": {
        "sms": "Responde STOP para cancelar.",
        "email": "Para dejar de recibir correos, haz clic aquí o responde STOP.",
    },
}


_NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*(?: [A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*)?$")
_AMENITY_RE = re.compile(r"^[A-Za-z0-9À-ÿ][A-Za-z0-9À-ÿ /&'\-]{0,39}$")
_INJECTION_WORDS = re.compile(
    r"\b(ignore|instruction|instructions|system|prompt|assistant|disregard|override|forget|pretend)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- reply intent


def classify_reply(text: str | None) -> ReplyIntent | None:
    """None when there is no reply; otherwise classifier.classify()."""
    if not text or not str(text).strip():
        return None
    return classify(str(text))


# ------------------------------------------------------------- pure helpers


def select_channel(preferences: list[str], consent: Consent) -> Channel | None:
    """First preferred sms/email channel with consent. Voice is skipped: it is
    never auto-sent (R1, R2; fall-through and empty-prefs are assumptions)."""
    order = [p for p in preferences] or list(DEFAULT_CHANNEL_ORDER)
    for pref in order:
        if pref in SENDABLE_CHANNELS and getattr(consent, f"{pref}_opt_in", False):
            return pref  # type: ignore[return-value]
    return None


def resolve_timezone(name: str | None) -> tuple[ZoneInfo, str | None]:
    """ZoneInfo for the record, or UTC with a note when missing/invalid."""
    if not name:
        return ZoneInfo("UTC"), "timezone missing, UTC assumed"
    try:
        return ZoneInfo(name), None
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC"), f"timezone {name!r} unknown, UTC assumed"


def compute_send_at(last_interaction_local: datetime, channel: str, delay_days: int) -> datetime:
    """last_interaction + delay, snapped to the next channel window at or after
    that instant. Same-day is allowed (R2: 05:30 + 3d -> 10:00 same day);
    a target past the window rolls to the next day (R1: 09:04 -> next 09:00)."""
    target = last_interaction_local + timedelta(days=delay_days)
    window = target.replace(hour=WINDOW_HOUR[channel], minute=0, second=0, microsecond=0)
    if target <= window:
        return window
    return window + timedelta(days=1)


def compute_horizon(send_date: date, move_date_target: date | None) -> Horizon:
    """Days from send date to move date; <= 60 short, else long (R1 32, R2 68)."""
    if move_date_target is None:
        return "unknown"
    return "short" if (move_date_target - send_date).days <= HORIZON_SHORT_MAX_DAYS else "long"


def delay_days(lifecycle_stage: str, horizon: Horizon, persona: str) -> int:
    """new -> 0 (R1). Otherwise the follow-up interval (R2: open + long -> 3)."""
    if lifecycle_stage == "new":
        return 0
    if persona == "resident":
        return RESIDENT_FOLLOW_UP_DAYS
    return FOLLOW_UP_DAYS[horizon]


def next_action_for(lifecycle_stage: str, horizon: Horizon, persona: str) -> dict:
    if persona == "resident":
        return {"type": "follow_up_in_days", "value": RESIDENT_FOLLOW_UP_DAYS}
    if lifecycle_stage == "new":
        label = "long" if horizon == "unknown" else horizon
        return {"type": "start_cadence", "name": f"prospect_welcome_{label}_horizon"}   # R1
    return {"type": "follow_up_in_days", "value": FOLLOW_UP_DAYS[horizon]}              # R2


def cta_type_for(primary_cta: str | None, persona: str) -> str:
    if primary_cta:
        key = primary_cta.strip().lower()
        return CTA_TYPE_MAP.get(key, key)
    return DEFAULT_CTA.get(persona, "contact_office")


def cta_link(property_name: str | None, cta_type: str) -> str:
    path = CTA_LINK_PATH.get(cta_type, cta_type.replace("_", "-"))
    return f"https://{property_slug(property_name)}.example/{path}"


def cta_for(cta_type: str, channel: str, property_name: str | None) -> Cta:
    """SMS tour CTA -> numeric options; everything else -> link (R1, R2; decision 3)."""
    if channel == "sms" and cta_type == "schedule_tour":
        return Cta(type=cta_type, options=list(TOUR_DAYS))
    return Cta(type=cta_type, link=cta_link(property_name, cta_type))


def property_short_name(property_name: str | None) -> str:
    """'Oak Ridge Apartments' -> 'Oak Ridge' (R1 body, R2 subject)."""
    if not property_name or not property_name.strip():
        return "our community"
    words = property_name.strip().split()
    while len(words) > 1 and words[-1].lower().strip(".,") in PROPERTY_SUFFIXES:
        words.pop()
    return " ".join(words)


def property_slug(property_name: str | None) -> str:
    """'Oak Ridge Apartments' -> 'oakridge' (R2 link)."""
    slug = re.sub(r"[^a-z0-9]", "", property_short_name(property_name).lower())
    return slug or "property"


def move_timeframe(move_date_target: date | None) -> str | None:
    """2026-02-15 -> 'mid-February' (R2). Days 1-10 early, 11-20 mid, 21+ late."""
    if move_date_target is None:
        return None
    part = "early" if move_date_target.day <= 10 else "mid" if move_date_target.day <= 20 else "late"
    return f"{part}-{move_date_target.strftime('%B')}"


def tour_days(send_at: datetime) -> tuple[list[str], str]:
    """Fixed Thu/Fri (R1); 'this week' for a Mon-Wed send, else 'next week'."""
    phrase = "this week" if send_at.weekday() <= 2 else "next week"
    return list(TOUR_DAYS), phrase


def sanitize_name(raw: object) -> str | None:
    """Only a plausible given name survives; anything instruction-like is dropped."""
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    if not name or len(name) > 30 or not _NAME_RE.match(name) or _INJECTION_WORDS.search(name):
        return None
    return name


def sanitize_amenities(raw: list[str] | None) -> list[str]:
    out: list[str] = []
    for item in raw or []:
        text = str(item).strip()
        if text and _AMENITY_RE.match(text) and not _INJECTION_WORDS.search(text):
            out.append(text)
    return out[:4]


def opt_out_line(channel: str, language: str) -> str:
    return OPT_OUT_LINES.get(language, OPT_OUT_LINES["en"])[channel]


def final_states(decision: Decision, validated: bool) -> list[str]:
    """states_verified for the output line; the two copy states need a passing draft."""
    states = list(decision.states_verified)
    if decision.send and validated:
        states.extend(s for s in POST_VALIDATION_STATES if s not in states)
    return states


# ---------------------------------------------------------------- resolve


def _no_send(record: Record, reason: str, next_action: dict, states: list[str] | None = None) -> Decision:
    return Decision(
        send=False,
        reason=reason,
        next_action=next_action,
        states_verified=list(states or []),
        task_id=record.task_id,
        persona=record.persona,
        language=record.input.language,
    )


def _inbound_reply(record: Record) -> str | None:
    for value in (record.inbound_reply, record.last_reply, record.input.inbound_reply, record.input.last_reply):
        if value and str(value).strip():
            return str(value)
    return None


def resolve(record: Record) -> Decision:
    stage = (record.lifecycle_stage or "new").strip().lower()
    persona = (record.persona or "prospect").strip().lower()
    inp = record.input

    # 1. lifecycle gates
    if stage == "do_not_contact":
        return _no_send(record, "do_not_contact", {"type": "mark_uncontactable"})
    if stage in ("closed", "leased"):
        return _no_send(record, f"lifecycle_{stage}", {"type": "none"})

    # 2. inbound reply gates
    intent = classify_reply(_inbound_reply(record))
    if intent == "stop":
        return _no_send(record, "opt_out_received", {"type": "mark_opted_out"})
    if intent == "question":
        return _no_send(record, "question_requires_agent", {"type": "assign_to_agent"})

    # 3. consent + channel
    channel = select_channel(record.channel_preferences, record.consent)
    if channel is None:
        if record.consent.voice_opt_in:
            return _no_send(record, "voice_requires_agent", {"type": "create_call_task"})
        return _no_send(record, "no_consented_channel", {"type": "mark_uncontactable"})
    states = list(POLICY_STATES)

    # 4. persona and required timing inputs
    if persona not in ("prospect", "resident"):
        return _no_send(record, "unsupported_persona", {"type": "flag_for_review"}, states)
    if inp.last_interaction is None:
        return _no_send(record, "missing_last_interaction", {"type": "flag_for_review"}, states)
    tz, tz_note = resolve_timezone(inp.timezone)
    li_local = inp.last_interaction.astimezone(tz)
    notes: list[str] = [tz_note] if tz_note else []

    cta_type = cta_type_for(record.assertions.constraints.primary_cta, persona)
    common = dict(
        task_id=record.task_id,
        persona=persona,
        channel=channel,
        states_verified=states,
        first_name=sanitize_name(inp.profile.first_name),
        property_short_name=property_short_name(inp.property_name),
        language=inp.language,
        amenities=sanitize_amenities(inp.profile.amenity_interest),
        opt_out_line=opt_out_line(channel, inp.language),
        max_chars=SMS_MAX_CHARS if channel == "sms" else None,
    )

    # 5a. resident: no horizon or tour logic
    if persona == "resident":
        send_at = compute_send_at(li_local, channel, delay_days(stage, "unknown", persona))
        reason = f"{channel} consented and preferred; resident {stage}"
        return Decision(
            send=True,
            reason="; ".join([reason, *notes]),
            send_at=send_at,
            horizon=None,
            cta=cta_for(cta_type, channel, inp.property_name),
            next_action=next_action_for(stage, "unknown", persona),
            **common,
        )

    # 5b. prospect replying to book a tour: confirm that day at the next window
    if intent in ("book_thu", "book_fri"):
        day = "Thu" if intent == "book_thu" else "Fri"
        send_at = compute_send_at(li_local, channel, 0)
        horizon = compute_horizon(send_at.date(), inp.move_date_target)
        return Decision(
            send=True,
            reason="; ".join([f"{channel} consented and preferred; booking reply for {day}", *notes]),
            send_at=send_at,
            horizon=horizon,
            cta=Cta(type="confirm_tour", options=[day]),
            next_action={"type": "book_tour", "day": day},
            message_kind="tour_confirmation",
            booked_day=day,
            move_timeframe=move_timeframe(inp.move_date_target),
            **common,
        )

    # 5c. prospect outreach. Horizon is measured from the send date, and for
    # `open` the delay depends on the horizon, so resolve in two passes; the
    # second pass only changes anything for move dates 60-63 days out.
    horizon = compute_horizon(li_local.date(), inp.move_date_target)
    send_at = compute_send_at(li_local, channel, delay_days(stage, horizon, persona))
    if inp.move_date_target is not None and inp.move_date_target < send_at.date():
        return _no_send(record, "stale_move_date", {"type": "flag_for_review"}, states)
    horizon2 = compute_horizon(send_at.date(), inp.move_date_target)
    if horizon2 != horizon:
        horizon = horizon2
        send_at = compute_send_at(li_local, channel, delay_days(stage, horizon, persona))
        horizon = compute_horizon(send_at.date(), inp.move_date_target)

    days, week_phrase = tour_days(send_at)
    days_out = (inp.move_date_target - send_at.date()).days if inp.move_date_target else None
    horizon_text = f"horizon {horizon}" + (f" ({days_out} days)" if days_out is not None else "")
    reason = f"{channel} consented and preferred; prospect {stage}; {horizon_text}"
    return Decision(
        send=True,
        reason="; ".join([reason, *notes]),
        send_at=send_at,
        horizon=horizon,
        cta=cta_for(cta_type, channel, inp.property_name),
        next_action=next_action_for(stage, horizon, persona),
        move_timeframe=move_timeframe(inp.move_date_target),
        tour_days=days,
        tour_week_phrase=week_phrase,
        **common,
    )
