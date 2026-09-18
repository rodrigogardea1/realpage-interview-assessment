"""Pydantic models for input records, the policy Decision, and the output line.

`Record` ignores unknown top-level keys, so the `expected` block never rides
along into policy or generation (CLAUDE.md: only evaluate.py reads it).
Date fields parse leniently: an unparseable value becomes None and policy
decides what to do with the gap.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

Channel = Literal["sms", "email", "voice"]
Horizon = Literal["short", "long", "unknown"]
MessageKind = Literal["outreach", "tour_confirmation"]


def _lenient_date(value: Any) -> date | None:
    if value is None or isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _lenient_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z") or text.endswith("z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:  # naive timestamps are taken as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------- input side


class Consent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    email_opt_in: bool = False
    sms_opt_in: bool = False
    voice_opt_in: bool = False


class Profile(BaseModel):
    model_config = ConfigDict(extra="allow")
    first_name: str | None = None
    amenity_interest: list[str] | None = None
    city_interest: str | None = None

    @field_validator("first_name", "city_interest", mode="before")
    @classmethod
    def _to_str(cls, v: Any) -> str | None:
        return None if v is None else str(v)

    @field_validator("amenity_interest", mode="before")
    @classmethod
    def _to_list(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
        return None


class RecordInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    property_name: str | None = None
    move_date_target: date | None = None
    last_interaction: datetime | None = None
    timezone: str | None = None
    language: str = "en"
    profile: Profile = Profile()
    inbound_reply: str | None = None
    last_reply: str | None = None

    @field_validator("move_date_target", mode="before")
    @classmethod
    def _date(cls, v: Any) -> date | None:
        return _lenient_date(v)

    @field_validator("last_interaction", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> datetime | None:
        return _lenient_datetime(v)

    @field_validator("language", mode="before")
    @classmethod
    def _lang(cls, v: Any) -> str:
        return "en" if not v else str(v).strip().lower()[:2]


class Constraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    no_pii_leak: bool = True
    no_sensitive_discrimination: bool = True
    include_opt_out_instructions: bool = True
    primary_cta: str | None = None


class Assertions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    required_states: list[str] = []
    constraints: Constraints = Constraints()


class Record(BaseModel):
    """One input line. `expected`, `thresholds`, and anything unknown are dropped."""

    model_config = ConfigDict(extra="ignore")
    task_id: str
    persona: str = "prospect"
    lifecycle_stage: str = "new"
    consent: Consent = Consent()
    channel_preferences: list[str] = []
    input: RecordInput = RecordInput()
    assertions: Assertions = Assertions()
    inbound_reply: str | None = None
    last_reply: str | None = None

    @field_validator("channel_preferences", mode="before")
    @classmethod
    def _prefs(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(x).strip().lower() for x in v if isinstance(x, str)]


# --------------------------------------------------------------- decision side


class Cta(BaseModel):
    type: str
    options: list[str] | None = None
    link: str | None = None

    def to_output(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class Decision(BaseModel):
    """Everything policy resolves. The generator sees this and nothing else."""

    send: bool
    reason: str
    next_action: dict[str, Any]
    states_verified: list[str] = []
    channel: Channel | None = None
    send_at: datetime | None = None
    horizon: Horizon | None = None
    cta: Cta | None = None
    message_kind: MessageKind = "outreach"
    # facts for the generator, all sanitized/derived deterministically
    task_id: str = ""
    persona: str = "prospect"
    first_name: str | None = None
    property_short_name: str | None = None
    language: str = "en"
    move_timeframe: str | None = None
    amenities: list[str] = []
    tour_days: list[str] | None = None
    tour_week_phrase: str | None = None
    booked_day: str | None = None
    opt_out_line: str = ""
    max_chars: int | None = None
    # Bounded, sanitized leftovers from the record. Copy-only: policy never reads it.
    extra_context: dict[str, str] = {}


# ----------------------------------------------------------------- output side


class Draft(BaseModel):
    subject: str | None = None
    body: str


class Validation(BaseModel):
    opt_out_present: bool
    pii_leak: bool
    fair_housing_flags: list[str]
    sms_length: int | None
    attempts: int


class NextMessage(BaseModel):
    channel: str
    send_at: str
    subject: str | None
    body: str
    cta: dict[str, Any]


class OutputLine(BaseModel):
    task_id: str
    decision: dict[str, Any]
    states_verified: list[str]
    next_message: NextMessage | None
    next_action: dict[str, Any]
    validation: Validation
    latency_ms: int
