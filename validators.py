"""Deterministic guardrails on a generated draft. No LLM calls, no I/O.

Each check returns a violation string (or a list for fair housing) and None
when clean. `run()` collects them; an empty list means the draft may ship.
Violation strings are stable prefixes (`pii_`, `fair_housing:`, `cta_`, ...)
so the agent and evaluate.py can classify failures by layer.
"""
from __future__ import annotations

import re
from typing import Any, Iterator

from models import Decision, Draft, Record, Validation

SMS_MAX_CHARS = 320
EMAIL_SUBJECT_MAX_CHARS = 120
MAX_EXCLAMATIONS = 1

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<![\d/])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\d/])")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Za-z]+\s+){1,3}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court|Pkwy|Parkway|Pl|Place|Ter|Terrace)\b\.?",
    re.IGNORECASE,
)
UNIT_RE = re.compile(r"\b(?:unit|apt\.?|apartment|suite|ste\.?)\s*#?\s*\d+[A-Za-z]?\b|#\s?\d{2,5}\b", re.IGNORECASE)
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
INJECTION_PHRASE_RE = re.compile(
    r"ignore (?:all |the |any |your )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)"
    r"|as an ai\b|system prompt|language model|you are now\b",
    re.IGNORECASE,
)

# Protected classes under the Fair Housing Act plus the banned phrases from
# CLAUDE.md. Describe the property, never the ideal tenant.
FAIR_HOUSING_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("familial_status", re.compile(
        r"\b(?:famil(?:y|ies)|kids?|child(?:ren)?|singles?|bachelors?|couples?|empty[- ]nesters?|adults?[- ]only"
        r"|familias?|ni[ñn][oa]s?)\b", re.IGNORECASE)),
    ("age", re.compile(
        r"\b(?:seniors?|elderly|retire[ed]s?|retirees?|mature|young|55\+|62\+|ancianos?|j[oó]venes|mayores)\b",
        re.IGNORECASE)),
    ("religion", re.compile(
        r"\b(?:church(?:es)?|temple|mosque|synagogue|christian|catholic|jewish|muslim|hindu|parish|congregation"
        r"|religio\w*|iglesia)\b", re.IGNORECASE)),
    ("race_national_origin", re.compile(
        r"\b(?:race|racial|ethnic\w*|hispanic|latin[oa]s?|asian|african|caucasian|immigrants?|nationality"
        r"|foreigners?|american[- ]born|english[- ]speaking|spanish[- ]speaking)\b", re.IGNORECASE)),
    ("sex", re.compile(r"\b(?:men|women|male|female|gender|ladies|gentlemen|guys)\b", re.IGNORECASE)),
    ("disability", re.compile(r"\b(?:disabled|disabilit\w*|handicap\w*|wheelchair|able-bodied)\b", re.IGNORECASE)),
    ("source_of_income", re.compile(
        r"\b(?:section 8|vouchers?|housing assistance|subsid\w*|income|employ\w*|professionals?|welfare|salary)\b",
        re.IGNORECASE)),
    ("ideal_tenant", re.compile(r"\b(?:perfect|ideal|great|suited|designed|made)\s+for\b", re.IGNORECASE)),
]
DAY_PATTERNS: dict[str, re.Pattern[str]] = {
    "Thu": re.compile(r"\b(?:thu(?:rs(?:day)?)?|jue(?:ves)?)\b", re.IGNORECASE),
    "Fri": re.compile(r"\b(?:fri(?:day)?|vie(?:rnes)?)\b", re.IGNORECASE),
}


# ---------------------------------------------------------------- checks


def check_opt_out(body: str, opt_out_line: str) -> str | None:
    """The exact channel/language opt-out line from policy must end the body (R1, R2)."""
    if not opt_out_line:
        return "opt_out_line_unresolved"
    if not body.rstrip().endswith(opt_out_line):
        return "opt_out_missing_or_not_last"
    return None


def _iter_strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _iter_strings(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _iter_strings(v, f"{path}[{i}]")


def check_pii(text: str, record: Record) -> str | None:
    """No emails, phones, SSNs, addresses, unit numbers, and no profile values
    beyond first_name and amenity_interest echoed into the copy."""
    if EMAIL_RE.search(text):
        return "pii_email"
    if SSN_RE.search(text):
        return "pii_ssn"
    if PHONE_RE.search(text):
        return "pii_phone"
    if ADDRESS_RE.search(text):
        return "pii_address"
    if UNIT_RE.search(text):
        return "pii_unit"
    profile = record.input.profile.model_dump()
    profile.pop("first_name", None)
    profile.pop("amenity_interest", None)
    lowered = text.lower()
    for path, value in _iter_strings(profile, "profile"):
        v = value.strip().lower()
        if len(v) >= 3 and v in lowered:
            return f"pii_profile_field:{path}"
    return None


def check_fair_housing(text: str) -> list[str]:
    flags: list[str] = []
    for label, pattern in FAIR_HOUSING_TERMS:
        m = pattern.search(text)
        if m:
            flags.append(f"fair_housing:{label}:{m.group(0).lower()}")
    return flags


def check_length(body: str, channel: str) -> str | None:
    if not body.strip():
        return "body_empty"
    if channel == "sms" and len(body) > SMS_MAX_CHARS:
        return f"sms_too_long:{len(body)}"
    return None


def check_subject(draft: Draft, channel: str) -> str | None:
    subject = (draft.subject or "").strip()
    if channel == "sms" and subject:
        return "subject_not_allowed_for_sms"
    if channel == "email":
        if not subject:
            return "subject_required_for_email"
        if len(subject) > EMAIL_SUBJECT_MAX_CHARS:
            return f"subject_too_long:{len(subject)}"
    return None


def check_cta(draft: Draft, decision: Decision) -> str | None:
    """Exactly one primary CTA matching decision.cta: link present (and no other
    links) for link CTAs; numbered options for SMS tour CTAs; no links on SMS."""
    cta = decision.cta
    if cta is None:
        return "cta_missing"
    text = f"{draft.subject or ''}\n{draft.body}"
    urls = [u.rstrip(".,;:") for u in URL_RE.findall(text)]
    if cta.link:
        if cta.link not in urls:
            return "cta_link_missing"
        if any(u != cta.link for u in urls):
            return "cta_extra_link"
        return None
    if urls:
        return "cta_unexpected_link"
    options = cta.options or []
    if not options:
        return "cta_options_empty"
    for i, option in enumerate(options, start=1):
        day_re = DAY_PATTERNS.get(option)
        if day_re is not None:
            if not day_re.search(draft.body):
                return f"cta_option_missing:{option}"
        elif option.lower() not in draft.body.lower():
            return f"cta_option_missing:{option}"
        if cta.type == "schedule_tour" and not re.search(rf"(?<!\d){i}(?!\d)", draft.body):
            return f"cta_reply_number_missing:{i}"
    return None


def check_personalization(draft: Draft, decision: Decision) -> str | None:
    text = f"{draft.subject or ''}\n{draft.body}".lower()
    if decision.first_name and decision.first_name.lower() not in text:
        return "personalization_missing_first_name"
    if decision.property_short_name and decision.property_short_name.lower() not in text:
        return "personalization_missing_property"
    return None


def check_injection(draft: Draft, record: Record, decision: Decision) -> str | None:
    """Untrusted input strings may not be echoed verbatim unless policy exposed
    them (name, property, amenities, language, timezone)."""
    text = f"{draft.subject or ''}\n{draft.body}".lower()
    if INJECTION_PHRASE_RE.search(text):
        return "injection_phrase"
    allowed = {
        v.strip().lower()
        for v in [decision.first_name, decision.property_short_name, record.input.property_name,
                  record.input.timezone, record.input.language, *decision.amenities]
        if isinstance(v, str) and v.strip()
    }
    sources: dict[str, Any] = record.input.model_dump()
    sources["inbound_reply"] = record.inbound_reply
    sources["last_reply"] = record.last_reply
    for path, value in _iter_strings(sources, "input"):
        v = value.strip().lower()
        if len(v) >= 8 and v not in allowed and v in text:
            return f"injection_echo:{path}"
    return None


def check_tone(body: str) -> str | None:
    if EMOJI_RE.search(body):
        return "tone_emoji"
    bangs = body.count("!")  # "¡" always pairs with "!", count closing marks only
    if bangs > MAX_EXCLAMATIONS:
        return f"tone_exclamations:{bangs}"
    return None


# ------------------------------------------------------------------- run


def run(draft: Draft, decision: Decision, record: Record) -> list[str]:
    channel = decision.channel or ""
    text = f"{draft.subject or ''}\n{draft.body}"
    violations: list[str] = []
    for v in (
        check_length(draft.body, channel),
        check_subject(draft, channel),
        check_opt_out(draft.body, decision.opt_out_line),
        check_pii(text, record),
        check_cta(draft, decision),
        check_personalization(draft, decision),
        check_injection(draft, record, decision),
        check_tone(draft.body),
    ):
        if v:
            violations.append(v)
    violations.extend(check_fair_housing(text))
    return violations


def summarize(draft: Draft | None, decision: Decision, violations: list[str], attempts: int) -> Validation:
    body = draft.body if draft else ""
    return Validation(
        opt_out_present=bool(draft) and not any(v.startswith("opt_out") for v in violations),
        pii_leak=any(v.startswith("pii_") for v in violations),
        fair_housing_flags=[v for v in violations if v.startswith("fair_housing:")],
        sms_length=len(body) if decision.channel == "sms" and draft else None,
        attempts=attempts,
    )
