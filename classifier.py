"""Inbound reply -> intent. Deterministic regexes, English and Spanish.

Intents: book_thu, book_fri, stop, question, other. Order of precedence:
opt-out first (compliance), then a single booked day, then a question.
A message naming both days is a question if it asks one, else other.
"""
from __future__ import annotations

import re
from typing import Literal

Intent = Literal["book_thu", "book_fri", "stop", "question", "other"]
INTENTS: tuple[Intent, ...] = ("book_thu", "book_fri", "stop", "question", "other")

# Whole-message opt-out keywords (carrier-style), optionally wrapped in "please".
_STOP_WORD_RE = re.compile(
    r"^\W*(?:please\s+)?(?:stop|stopall|stop all|end|quit|cancel|unsubscribe|alto|parar|cancelar|basta)"
    r"(?:\s+please|\s+por favor)?\W*$",
    re.IGNORECASE,
)
# Opt-out phrases inside longer messages. "stop by" is excluded on purpose.
_STOP_PHRASE_RE = re.compile(
    r"\b(?:unsubscribe|opt[\s-]?out|remove me|take me off|leave me alone"
    r"|stop\s+(?:texting|messaging|contacting|emailing|calling|sending|it|now|this|these|all|please)"
    r"|(?:don'?t|do not|never|no)\s+(?:text|message|contact|email|call)\b"
    r"|no me (?:escriban|escribas|contacten|contactes|manden|mandes|llamen|llames)"
    r"|d[ée]j(?:en|a)me de|no quiero (?:m[aá]s )?mensajes|cancelar suscripci[oó]n)",
    re.IGNORECASE,
)
_THU_RE = re.compile(r"(?:^\W*(?:option\s*|#|opci[oó]n\s*)?1\b)|\b(?:thu(?:rs(?:day)?)?|jueves)\b", re.IGNORECASE)
_FRI_RE = re.compile(r"(?:^\W*(?:option\s*|#|opci[oó]n\s*)?2\b)|\b(?:fri(?:day)?|viernes)\b", re.IGNORECASE)
_QUESTION_RE = re.compile(
    r"[?¿]|^\s*(?:what|when|where|how|why|which|is|are|do|does|did|can|could|would|will|any|"
    r"qu[eé]|cu[aá]ndo|cu[aá]nto|cu[aá]l|d[oó]nde|c[oó]mo|tienen|hay|puedo|pueden)\b",
    re.IGNORECASE,
)


def is_opt_out(text: str) -> bool:
    return bool(_STOP_WORD_RE.search(text) or _STOP_PHRASE_RE.search(text))


def classify(text: str | None) -> Intent:
    t = (text or "").strip()
    if not t:
        return "other"
    if is_opt_out(t):
        return "stop"
    thu, fri = bool(_THU_RE.search(t)), bool(_FRI_RE.search(t))
    if thu and fri:
        return "question" if _QUESTION_RE.search(t) else "other"
    if thu:
        return "book_thu"
    if fri:
        return "book_fri"
    if _QUESTION_RE.search(t):
        return "question"
    return "other"
