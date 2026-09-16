"""The only place that talks to a model. `complete(prompt, schema, *, task_id)`.

Provider from env: LLM_PROVIDER (anthropic | stub), LLM_API_KEY (falls back to
ANTHROPIC_API_KEY), LLM_MODEL. A `.env` in the working directory is loaded
first without overriding real environment variables. Every prompt/response
pair is appended to logs/<task_id>.jsonl (LLM_LOG_DIR overrides the folder).

The `stub` provider renders a canned draft from the <facts> block in the
prompt so the whole pipeline runs offline in tests. LLM_STUB_RESPONSE (a JSON
string) overrides the stub output for failure-injection tests.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = {"anthropic": "claude-opus-5", "stub": "stub"}
DEFAULT_TEMPERATURE = 0.2
MAX_TOKENS = 8000  # adaptive thinking counts against max_tokens
SYSTEM_PROMPT = "You write short, compliant outbound messages for apartment leasing teams. Return only JSON."


class LLMError(RuntimeError):
    """Any provider failure. The agent turns this into generation_failed."""


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def settings() -> dict[str, Any]:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    return {
        "provider": provider,
        "model": os.getenv("LLM_MODEL") or DEFAULT_MODEL.get(provider, ""),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
        "effort": os.getenv("LLM_EFFORT", "low"),
    }


_NO_SAMPLING = re.compile(r"opus-4-[78]|opus-5|sonnet-5|fable|mythos")
_HAS_EFFORT = re.compile(r"opus-4-[5-8]|sonnet-4-6|opus-5|sonnet-5|fable|mythos")


def _supports_temperature(model: str) -> bool:
    # Sampling params were removed on Opus 4.7/4.8, Opus 5, Sonnet 5 and Fable (400 if sent).
    return not _NO_SAMPLING.search(model)


def _supports_effort(model: str) -> bool:
    # `output_config.effort` exists from Opus 4.5 onward; Sonnet 4.5 / Haiku 4.5 and older 400 on it.
    return bool(_HAS_EFFORT.search(model))


# ---------------------------------------------------------------- providers


_CLIENTS: dict[str | None, Any] = {}


def _anthropic_client(api_key: str | None) -> Any:
    """One client per key for the life of the process, so connections are reused."""
    import anthropic

    if api_key not in _CLIENTS:
        _CLIENTS[api_key] = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    return _CLIENTS[api_key]


def _anthropic(prompt: str, schema: dict, s: dict[str, Any]) -> dict:
    import anthropic

    client = _anthropic_client(s["api_key"])
    # The static system block carries a cache marker. Whether an entry is
    # actually created depends on the model's minimum cacheable prefix
    # (Haiku 4.5: 4096 tokens, Sonnet 4.5: 1024); shorter prefixes are a no-op.
    system_text = s.get("system") or SYSTEM_PROMPT
    kwargs: dict[str, Any] = dict(
        model=s["model"],
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    if _supports_effort(s["model"]):
        kwargs["output_config"]["effort"] = s["effort"]
    if _supports_temperature(s["model"]):
        # anthropic>=1.0 dropped sampling params from the typed signature; the wire field still exists.
        kwargs["extra_body"] = {"temperature": s["temperature"]}
    try:
        response = client.messages.create(**kwargs)
    except anthropic.APIError as e:  # auth, rate limit, 5xx, connection: all fail closed upstream
        raise LLMError(f"{type(e).__name__}: {e}") from e
    usage = getattr(response, "usage", None)
    s["_usage"] = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }
    if response.stop_reason == "refusal":
        raise LLMError("model refused the request")
    if response.stop_reason == "max_tokens":
        raise LLMError("response truncated at max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"non-JSON response: {text[:200]!r}") from e


def _stub(prompt: str, schema: dict, s: dict[str, Any]) -> dict:
    override = os.getenv("LLM_STUB_RESPONSE")
    if override:
        return json.loads(override)
    try:
        facts = json.loads(prompt.split("<facts>", 1)[1].split("</facts>", 1)[0])
    except (IndexError, json.JSONDecodeError) as e:
        raise LLMError("stub could not find a <facts> block in the prompt") from e
    name = facts.get("first_name") or "there"
    prop = facts.get("property_short_name") or "our community"
    opt_out = facts["opt_out_line"]
    cta = facts.get("cta") or {}
    week = facts.get("tour_week_phrase") or "this week"
    timeframe = facts.get("move_timeframe")
    amenities = facts.get("amenities") or []
    amenity_text = " and ".join(amenities) if len(amenities) <= 2 else ", ".join(amenities[:-1]) + f", and {amenities[-1]}"
    if facts.get("channel") == "sms":
        if cta.get("type") == "confirm_tour":
            day = {"Thu": "Thursday", "Fri": "Friday"}.get((cta.get("options") or ["Thu"])[0], "Thursday")
            body = f"Hi {name}—you're set for a tour of {prop} on {day}. We'll text you a time shortly. {opt_out}"
        elif cta.get("options"):
            body = (f"Hi {name}—welcome to {prop}! Tours are available {week}. Would you like to book a time on "
                    f"Thursday or Friday? Reply 1 for Thu, 2 for Fri. {opt_out}")
        else:
            body = f"Hi {name}—a quick note from {prop}. When you have a moment: {cta.get('link', '')} {opt_out}"
        return {"subject": "", "body": body}
    subject = f"Tour {prop}" + (f"—see the {amenity_text} you asked about" if amenities else " this week")
    lead = f"Since you're planning a {timeframe} move, " if timeframe else ""
    look = f"here's a quick look at our {amenity_text}." if amenities else "here's a quick look at what we offer."
    verb = "Book" if cta.get("type") in ("schedule_tour", None) else "Go"
    body = (f"Hi {name},\n{lead}{look} Book a visit {week} to compare floor plans.\n"
            f"{verb} now → {cta.get('link', '')}\n{opt_out}")
    return {"subject": subject, "body": body}


PROVIDERS: dict[str, Callable[[str, dict, dict[str, Any]], dict]] = {"anthropic": _anthropic, "stub": _stub}


# --------------------------------------------------------------- entry point


def _log(task_id: str, entry: dict[str, Any]) -> None:
    log_dir = Path(os.getenv("LLM_LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id) or "unknown"
        with (log_dir / f"{safe_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never take down a run


def complete(prompt: str, schema: dict, *, task_id: str, system: str | None = None) -> dict:
    """`system` is the static, cacheable part of the prompt; `prompt` is the per-call user turn."""
    s = settings()
    s["system"] = system
    provider = PROVIDERS.get(s["provider"])
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "provider": s["provider"],
        "model": s["model"],
        "system": system,
        "prompt": prompt,
        "response": None,
        "usage": None,
        "error": None,
        "latency_ms": None,
    }
    start = time.perf_counter()
    try:
        if provider is None:
            raise LLMError(f"unknown LLM_PROVIDER {s['provider']!r}")
        data = provider(prompt, schema, s)
        if not isinstance(data, dict):
            raise LLMError("provider returned non-object JSON")
        entry["response"] = data
        return data
    except LLMError as e:
        entry["error"] = str(e)
        raise
    except Exception as e:  # noqa: BLE001 - one boundary, one error type
        entry["error"] = f"{type(e).__name__}: {e}"
        raise LLMError(entry["error"]) from e
    finally:
        entry["latency_ms"] = int((time.perf_counter() - start) * 1000)
        entry["usage"] = s.get("_usage")
        _log(task_id, entry)
