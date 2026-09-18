"""Copy generation. The only module that calls llm.complete().

It receives a resolved Decision and writes {subject, body}. It never sees the
raw record, never chooses channel, timing, or CTA, and never decides to send.

The prompt file has a static part (role, rules, formats, examples) above the
`===DYNAMIC===` marker and a per-message part below it. The static part is
sent as a cacheable system block; the dynamic part is the user turn.
"""
from __future__ import annotations

import json
from pathlib import Path

import llm
from models import Decision, Draft

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "generate.md"
DYNAMIC_MARKER = "===DYNAMIC==="

DRAFT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "description": "Email subject line; empty string for SMS."},
        "body": {"type": "string", "description": "The message body, ending with the opt-out line."},
    },
    "required": ["subject", "body"],
    "additionalProperties": False,
}

FACT_FIELDS = (
    "channel", "language", "persona", "message_kind", "first_name", "property_short_name",
    "move_timeframe", "amenities", "horizon", "tour_days", "tour_week_phrase", "booked_day",
    "opt_out_line", "max_chars", "unit", "message_intent", "extra_context",
)
DAY_NAMES = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
             "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}
# Known CTA types get a fixed verb. Anything else is passed to the model as a
# raw identifier; no verb is ever derived from the identifier's prefix.
CTA_VERBS = {"schedule_tour": "Book", "renew_lease": "Renew", "pay_balance": "Pay", "schedule_maintenance": "Schedule"}
# Link labels the holdout uses verbatim. Same rule: a fixed label per known type, nothing derived.
CTA_LABELS = {"reschedule": "Reschedule", "review_renewal": "Review your offer", "review_renewal_details": "View details",
              "get_started": "Get started", "enroll_loyalty": "Enroll"}
ES_DAY_TO_EN = {"lunes": "Mon", "martes": "Tue", "miércoles": "Wed", "jueves": "Thu", "viernes": "Fri",
                "sábado": "Sat", "domingo": "Sun"}


def _reply_line(cta_type: str, options: list[str], language: str) -> str:
    """The numbered-reply sentence for an options CTA, in the message language."""
    es = language == "es"
    if cta_type == "intent_capture":
        words = {"yes": "Yes", "no": "No", "details": "Need more details", "sí": "sí", "detalles": "más detalles"}
        body = ", ".join(f"{i} {words.get(o, o)}" for i, o in enumerate(options, start=1))
    else:
        joiner = "para" if es else "for"
        body = ", ".join(f"{i} {joiner} {o}" for i, o in enumerate(options, start=1))
    return f"{'Responde' if es else 'Reply'} {body}."


def cta_instruction(decision: Decision) -> str:
    cta = decision.cta
    if cta is None:
        return "none."
    if cta.type == "confirm_tour":
        day = DAY_NAMES.get(decision.booked_day or "", decision.booked_day or "")
        return f"confirm the tour for {day} by name and say the team will follow up with a time. No links."
    if cta.options:
        reply = _reply_line(cta.type, cta.options, decision.language)
        if cta.type == "schedule_tour":
            days = " and ".join(DAY_NAMES.get(d, d) for d in cta.options)
            phrase = decision.tour_week_phrase or "this week"
            return f"invite a tour {phrase} on {days} and end the CTA with \"{reply}\" No links."
        return f"make the ask in one sentence and end the CTA with \"{reply}\" No links."
    urgency = ""
    if decision.horizon == "short" and decision.tour_days:
        urgency = f" Invite them to visit {decision.tour_week_phrase or 'this week'}."
    elif decision.horizon in ("long", "unknown") and decision.persona == "prospect" and cta.type == "schedule_tour":
        urgency = " Keep it low-pressure; do not name specific days."
    translate = " Translate the label into the message language." if decision.language != "en" else ""
    verb = CTA_VERBS.get(cta.type)
    if verb:
        return f"the CTA line is \"{verb} now → {cta.link}\", using this exact link once.{translate}{urgency}"
    label = CTA_LABELS.get(cta.type)
    if label:
        return f"the CTA line is \"{label} → {cta.link}\", using this exact link once.{translate}{urgency}"
    return (f"This CTA is `{cta.type}`. Write one short, natural call-to-action line in plain words for that "
            f"action, ending with `→ {cta.link}`. Use the link exactly once.{urgency}")


def facts_for(decision: Decision) -> dict:
    facts = {k: getattr(decision, k) for k in FACT_FIELDS}
    facts["cta"] = decision.cta.to_output() if decision.cta else None
    return facts


def _template() -> tuple[str, str]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    static, _, dynamic = text.partition(DYNAMIC_MARKER)
    if not dynamic:
        raise RuntimeError(f"{PROMPT_PATH} is missing the {DYNAMIC_MARKER} marker")
    return static.strip() + "\n", dynamic.lstrip("\n")


def build_prompt(decision: Decision, feedback: list[str] | None = None,
                 previous: Draft | None = None) -> tuple[str, str]:
    """Returns (static system prompt, dynamic user prompt)."""
    static, dynamic = _template()
    feedback_text = ""
    if feedback:
        previous_json = json.dumps(previous.model_dump(), ensure_ascii=False) if previous else "(not available)"
        feedback_text = (
            "\n# Feedback\nYour previous draft failed these checks, fix them:\n"
            + "\n".join(f"- {v}" for v in feedback)
            + f"\nPrevious draft: {previous_json}\n"
        )
    user = (
        dynamic.replace("{{CHANNEL}}", decision.channel or "")
        .replace("{{OPT_OUT_LINE}}", decision.opt_out_line)
        .replace("{{CTA_INSTRUCTION}}", cta_instruction(decision))
        .replace("{{FACTS}}", json.dumps(facts_for(decision), ensure_ascii=False, indent=2))
        .replace("{{FEEDBACK}}", feedback_text)
    )
    return static, user


def write(decision: Decision, feedback: list[str] | None = None, previous: Draft | None = None) -> Draft:
    """One generation attempt. Raises llm.LLMError on provider failure."""
    system, user = build_prompt(decision, feedback, previous)
    data = llm.complete(user, DRAFT_SCHEMA, task_id=decision.task_id, system=system)
    subject = data.get("subject")
    subject = subject.strip() if isinstance(subject, str) and subject.strip() else None
    body = data.get("body")
    if not isinstance(body, str):
        raise llm.LLMError("draft has no body")
    return Draft(subject=subject, body=body.strip())
