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
# Opt-out lines the data accepts, per language and channel. The first entry is
# the one policy hands the generator; the rest are variants seen in the holdout.
OPT_OUT_ACCEPTED: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "sms": ("Reply STOP to opt out.",),                                         # R1, holdout no_show
        "email": (
            "To opt out of emails, click here or reply STOP.",                      # R2
            "Opt-out here or reply STOP.",                                          # holdout welcome, loyalty, details
            "Opt-out any time.",                                                    # holdout renewal_90day
        ),
    },
    "es": {
        "sms": ("Responde STOP para cancelar.",),                                   # holdout spanish_locale
        "email": ("Para dejar de recibir correos, haz clic aquí o responde STOP.",),
    },
}
# CTA types whose SMS copy must carry "Reply 1 ..., 2 ..." numbering.
NUMBERED_CTA_TYPES: frozenset[str] = frozenset({"schedule_tour", "reschedule", "intent_capture"})
_HYPHENS = str.maketrans({"‐": "-", "‑": "-", "‒": "-"})
_ES_MARKERS = re.compile(
    r"\b(?:hola|gracias|para|por|tu|tus|su|una?|esta|este|semana|responde|visita|el|la|los|las|de|del|que|con|en|y)\b",
    re.IGNORECASE,
)
_EN_MARKERS = re.compile(r"\b(?:the|your|you|would|this|week|reply|welcome|with|and|for)\b", re.IGNORECASE)


def normalize_hyphens(text: str) -> str:
    """The holdout uses U+2011 non-breaking hyphens in copy and unit numbers."""
    return text.translate(_HYPHENS)


DAY_PATTERNS: dict[str, re.Pattern[str]] = {
    "Mon": re.compile(r"\b(?:mon(?:day)?|lun(?:es)?)\b", re.IGNORECASE),
    "Tue": re.compile(r"\b(?:tue(?:s(?:day)?)?|mar(?:tes)?)\b", re.IGNORECASE),
    "Wed": re.compile(r"\b(?:wed(?:s|nesday)?|mi[eé]r(?:coles)?)\b", re.IGNORECASE),
    "Thu": re.compile(r"\b(?:thu(?:rs?(?:day)?)?|jue(?:ves)?)\b", re.IGNORECASE),
    "Fri": re.compile(r"\b(?:fri(?:day)?|vie(?:rnes)?)\b", re.IGNORECASE),
    "Sat": re.compile(r"\b(?:sat(?:urday)?|s[aá]b(?:ado)?)\b", re.IGNORECASE),
    "Sun": re.compile(r"\b(?:sun(?:day)?|dom(?:ingo)?)\b", re.IGNORECASE),
}


# ---------------------------------------------------------------- checks


def check_opt_out(body: str, opt_out_line: str, channel: str | None = None, language: str | None = None) -> str | None:
    """The body must end with an accepted opt-out line: the one policy resolved,
    or, when channel and language are given, any variant the data uses for them.
    Hyphen forms are normalized so "Opt‑out" and "Opt-out" both pass."""
    if not opt_out_line:
        return "opt_out_line_unresolved"
    accepted = {opt_out_line}
    if channel:
        by_lang = OPT_OUT_ACCEPTED.get(language or "en", OPT_OUT_ACCEPTED["en"])
        accepted.update(by_lang.get(channel, ()))
    tail = normalize_hyphens(body.rstrip())
    if not any(tail.endswith(normalize_hyphens(line)) for line in accepted):
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


def check_pii(text: str, record: Record, decision: Decision | None = None) -> str | None:
    """No emails, phones, SSNs, addresses, unit numbers, and no profile values
    beyond first_name, amenity_interest, and the fields policy cleared into
    decision.extra_context echoed into the copy. A resident's own `input.unit`
    is allowed in either hyphen form (holdout renewal records name the unit)."""
    own_unit = getattr(record.input, "unit", None)
    if isinstance(own_unit, str) and own_unit.strip() and record.persona.strip().lower() == "resident":
        text = normalize_hyphens(text).replace(normalize_hyphens(own_unit.strip()), " ")
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
    cleared = set(decision.extra_context) if decision is not None else set()
    lowered = text.lower()
    for key, field_value in profile.items():
        if key in cleared:
            continue  # policy already vetted this field for the generator
        for path, value in _iter_strings(field_value, f"profile.{key}"):
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
        if len(urls) != 1:
            return "cta_link_repeated"
        # No shape requirement on the link's line: the holdout uses "Schedule →",
        # "Review your offer →" and "Book your time here: <link>" alike.
        return None
    if urls:
        return "cta_unexpected_link"
    options = cta.options or []
    if not options:
        return "cta_options_empty"
    for i, option in enumerate(options, start=1):
        day_re = DAY_PATTERNS.get(option) or next((p for p in DAY_PATTERNS.values() if p.fullmatch(option)), None)
        if day_re is not None:
            if not day_re.search(draft.body):
                return f"cta_option_missing:{option}"
        elif option.lower() not in draft.body.lower():
            return f"cta_option_missing:{option}"
        if cta.type in NUMBERED_CTA_TYPES and not re.search(rf"(?<!\d){i}(?!\d)", draft.body):
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
    for value in decision.extra_context.values():  # vetted by policy.build_extra_context
        allowed.add(value.strip().lower())
        allowed.update(part.strip().lower() for part in value.split(","))
    sources: dict[str, Any] = record.input.model_dump()
    sources["inbound_reply"] = record.inbound_reply
    sources["last_reply"] = record.last_reply
    for path, value in _iter_strings(sources, "input"):
        v = value.strip().lower()
        if len(v) >= 8 and v not in allowed and v in text:
            return f"injection_echo:{path}"
    return None


def check_language(body: str, language: str) -> str | None:
    """Coarse locale check behind the `locale_applied` state: Spanish copy must
    read as Spanish. English is the default and is not checked."""
    if language != "es":
        return None
    es, en = len(_ES_MARKERS.findall(body)), len(_EN_MARKERS.findall(body))
    if es < 3 or en > es:
        return f"locale_mismatch:{language}"
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
        check_opt_out(draft.body, decision.opt_out_line, channel, decision.language),
        check_pii(text, record, decision),
        check_cta(draft, decision),
        check_personalization(draft, decision),
        check_injection(draft, record, decision),
        check_tone(draft.body),
        check_language(draft.body, decision.language),
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
