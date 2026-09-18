# Design note: per-brand style config

**Status:** proposed, not built. Today the brand voice is hardcoded to one property (Oak Ridge), inferred from the two sample messages. This note describes how it becomes config so the same agent serves many properties under different owners.

## Why

RealPage serves thousands of properties. Each owner has its own voice, opt-out wording, legal footer, and channel rules. Style can't live in code or in a single prompt file. It also can't be free-form: the validators need to know what "on brand" means in order to check it, and `brand_style_applied` should only be claimed when a concrete config was actually enforced.

## Shape

One JSON (or YAML) file per brand, resolved by property. Example `brands/oak-ridge.json`:

```json
{
  "brand_id": "oak-ridge",
  "properties": ["Oak Ridge Apartments"],
  "tone": ["warm", "brief", "concrete"],
  "greeting": {"sms": "Hi {first_name}—", "email": "Hi {first_name},"},
  "fallback_name": "there",
  "sign_off": null,
  "max_exclamations": 1,
  "emoji": false,
  "sms": {"soft_max_chars": 160, "hard_max_chars": 320},
  "email": {"subject_max_chars": 80, "cta_line": "{verb} now → {link}"},
  "opt_out": {
    "en": {"sms": "Reply STOP to opt out.", "email": "To opt out of emails, click here or reply STOP."},
    "es": {"sms": "Responde STOP para cancelar.", "email": "Para dejar de recibir correos, haz clic aquí o responde STOP."}
  },
  "send_windows": {"sms": "09:00", "email": "10:00"},
  "quiet_hours": {"start": "21:00", "end": "08:00"},
  "tour_days_default": ["Thu", "Fri"],
  "cta_catalog": {
    "schedule_tour": {"verb": "Book", "url": "https://oakridge.example/tour"},
    "renew_lease":   {"verb": "Renew", "url": "https://oakridge.example/renew"}
  },
  "banned_phrases": ["luxury living", "exclusive"],
  "style_examples": {"sms": ["..."], "email": ["..."]}
}
```

A `brands/default.json` carries the values currently hardcoded, so behavior is unchanged for records with no matching brand.

## Where each field lands

| Config field | Consumed by | Replaces |
|---|---|---|
| `send_windows`, `quiet_hours`, `tour_days_default` | `policy.py` | `WINDOW_HOUR`, `TOUR_DAYS` constants |
| `opt_out` | `policy.py` (`opt_out_line`) | `OPT_OUT_LINES` |
| `cta_catalog` | `policy.py` (`cta_for`, `cta_link`) | `CTA_TYPE_MAP`, `CTA_LINK_PATH`, string-built `.example` URLs |
| `greeting`, `tone`, `sign_off`, `emoji`, `style_examples` | `generator.py` → dynamic section of `prompts/generate.md` | Hardcoded format rules and the two few-shot messages |
| `max_exclamations`, `emoji`, `sms.*`, `email.*`, `banned_phrases` | `validators.py` (`check_tone`, `check_length`, `check_subject`, new `check_banned`) | Module constants |
| `cta_catalog[...].verb` | `generator.py` (`cta_instruction`) | `CTA_VERBS` |

Resolution: `brands.load(property_name) -> Brand` in a new `brands.py`, called once in `policy.resolve()`; the resolved `Brand` rides on `Decision` so generator and validators read the same object. Records with no matching property get `default`.

## What stays out of config

- Consent gating, lifecycle gates, horizon logic, fail-closed behavior. Those are policy, not style, and are the same for every brand.
- Fair-housing term list. Brands can add banned phrases; they cannot remove protected-class checks.
- The PII and injection rules.

## `brand_style_applied` under this design

Claimed only when: a `Brand` was resolved (default counts), the prompt was built from it, and the brand-derived validators passed. The state means "this specific config was enforced," not "the copy looked fine."

## Migration path

1. `brands.py` + `brands/default.json` with today's constants. No behavior change; tests unchanged.
2. Move constants out of `policy.py` / `validators.py` / `generator.py` into the loader. Tests unchanged.
3. Add `brands/oak-ridge.json` and a second fixture brand with different opt-out text and no em-dash greeting; add tests that the same record produces brand-appropriate copy under each.
4. Later: a per-brand fine-tuned adapter or retrieval over approved messages, keyed by `brand_id`.

## Open questions for product

- Is brand keyed by property, by owner/management company, or both (owner default with property overrides)?
- Do legal footers vary by state as well as by brand?
- Who owns the config — is it edited by marketing in a UI, or by engineering in the repo?
