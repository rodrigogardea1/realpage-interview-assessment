# Outreach Agent

A context-aware message-sending agent for multifamily leasing. Given one record describing a prospect or resident — consent, channel preferences, lifecycle stage, move date, profile — it decides **whether** to contact them, **which channel**, **when**, and **what to say**, and emits one JSON line. It sits between a CRM and a delivery provider; it decides, it does not send.

Outputs are graded on semantic match to an `expected` block, so the design goal is: every decision with a single right answer lives in deterministic code, and the LLM writes only the copy, behind validators.

## Architecture

```
record ──► policy.resolve() ──► Decision ──► generator.write() ──► Draft
              (no LLM)             │  (send=False → skip)      │
                                   │                       validators.run()
                                   │                            │
                                   │              fail → retry ≤2 with feedback
                                   │                       then fail closed
                                   ▼                            ▼
                             OutputLine (task_id, decision, states_verified,
                                         next_message, next_action, validation, latency_ms)
```

| Module | Role |
|---|---|
| `policy.py` | Pure functions. Consent gating, channel selection, send window, horizon, cadence, CTA shape. No I/O, no model. |
| `generator.py` + `prompts/generate.md` | The only model call. Receives the resolved `Decision`, never the raw record. Sample expected messages used as few-shot. |
| `validators.py` | Opt-out line, PII, fair-housing terms, SMS length, CTA present, personalization, injection echo, tone. Returns violations. |
| `classifier.py` | Inbound reply → `book_thu / book_fri / stop / question / other`. Regex-first. |
| `agent.py` | Orchestration, retry loop, fail-closed, CLI. |
| `evaluate.py` + `prompts/judge.md` | Scorecard: hard-field diffs, semantic body score, safety count, p95 latency, reply F1. |
| `models.py` | Pydantic models. `Record` ignores unknown keys, so `expected` is dropped at parse time and can never leak into policy or generation. |

**Why the split.** Consent, timing, and fair-housing exposure are compliance decisions (TCPA, CAN-SPAM, FHA). Those need to be unit-testable, diffable, and explainable to a compliance reviewer. Copy is the one open-ended decision, so it goes to the model — with deterministic checks on the way out.

## Rules extracted from `sample.jsonl`

Only two records were provided, so every rule cites its evidence. Rules marked *assumption* are judgment calls where the data underdetermines the answer.

| Rule | Evidence |
|---|---|
| Channel = first `channel_preferences` entry whose consent flag is true. No consented channel → no send. | R1 (sms preferred + consented → sms), R2 (email preferred + consented → email). Fall-through is inferred, not evidenced. |
| Send windows: SMS 09:00, email 10:00, local to `input.timezone`. | R1, R2 |
| `send_at` = `last_interaction` + delay, snapped to the next window at or after that instant. Same-day send allowed if the target lands before the window. | R1: Mon 09:04 + 0d → past 09:00 → Tue 09:00. R2: Sat 05:30 + 3d → Tue 05:30 → same day 10:00. Both exact. |
| Lifecycle `new` → `start_cadence prospect_welcome_<horizon>_horizon`, delay 0. Lifecycle `open` → `follow_up_in_days` (3 long / 2 short), delay = that value. | R1 (new), R2 (open). *Assumption:* lifecycle and horizon are confounded in the sample; this split reproduces both records and is the more defensible reading. |
| Horizon = days from send date to `move_date_target`. ≤ 60 short, else long, missing → long. | R1 32d short, R2 68d long. *Assumption:* threshold anywhere in 33–67 fits; 60 is the common leasing convention. |
| CTA type from `primary_cta` (`book_tour` → `schedule_tour`). Shape follows channel: SMS → numeric reply options for two named days; email → booking link. | R1, R2. *Assumption:* shape could equally follow horizon; channel is the simpler rule. |
| SMS: no subject, one line, "Reply STOP to opt out." last, ≤ 320 chars hard. Email: subject, greeting on its own line, link CTA line, "To opt out of emails, click here or reply STOP." last. | R1, R2. Sample SMS is 166 chars, so 160 is a soft target. |
| Personalize with first name, property short name, move timeframe (early/mid/late Month), amenity interests. Not every profile field is used (R1 `city_interest` is absent from expected copy). | R2 subject/body, R1 |
| Never invent amenity details. Expected R2 says "24/7 fitness center"; input says "fitness". We describe amenities only as named. | Deliberate deviation from R2. |
| `required_states` must be a subset of emitted `states_verified`. Policy asserts `consent_verified`; the copy states are added only after validators pass. No-send lines never claim checks that did not run. | R1, R2 |

### No-send cases (checked in this order)

| Condition | reason | next_action |
|---|---|---|
| `lifecycle_stage` do_not_contact | `do_not_contact` | `mark_uncontactable` |
| Inbound reply classified STOP | `opt_out_received` | `mark_opted_out` |
| Inbound reply classified question | `question_requires_agent` | `assign_to_agent` |
| `lifecycle_stage` closed / leased | `lifecycle_<stage>` | `none` |
| No preferred channel consented | `no_consented_channel` | `mark_uncontactable` |
| Only voice consented | `voice_requires_agent` | `create_call_task` |
| `move_date_target` in the past | `stale_move_date` | `flag_for_review` |
| Generation or validation failed after retries | `generation_failed` / `validation_failed` | `flag_for_review` |

## Guardrails on generated copy

- **Fair housing** — no reference to race, color, religion, national origin, sex, familial status, disability, age, or source of income. Property and amenities only, never the ideal tenant.
- **PII** — first name and property only. Any other profile value echoed in the body fails validation.
- **Injection** — profile fields are untrusted. Names are sanitized in policy; the generator never sees the raw record; bodies that echo untrusted input verbatim are rejected.
- **Opt-out** — always present, channel- and language-appropriate, literal `STOP` keyword kept in Spanish.
- **Fail closed** — any unresolved violation or model error produces `send: false`, never an unvalidated message.

## Thresholds and how they're measured

| Threshold | Measured by |
|---|---|
| `p95_latency_ms` 2000 | `evaluate.py`, wall clock per record. Model choice: `claude-haiku-4-5` default (~X ms); `claude-sonnet-4-5` ~3–4 s. *(fill in from your run)* |
| `personalization_score_min` | LLM judge (`prompts/judge.md`), 0–1, plus deterministic check that available personalization fields appear. |
| `reply_classification_f1_min` 0.9 | Macro F1 of `classifier.py` on `tests/replies.jsonl` (~30 labeled replies). *(fill in score)* |
| `safety_violations_max` 0 | Count of validator failures surviving retries. |

## Running it

```bash
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY

pytest -q                       # 150+ tests, offline via stub provider
python agent.py --in sample.jsonl --out out/sample_out.jsonl
python evaluate.py --in sample.jsonl
python agent.py --in tests/edge_cases.jsonl --out out/edge_out.jsonl

# hold-out export
python agent.py --in holdout.jsonl --out out/holdout_out.jsonl
```

Every prompt and model response is logged to `logs/<task_id>.jsonl`.

## What I'd do next

- **Confirm the ambiguous rules with product**: the 60-day horizon, lifecycle-vs-horizon split, same-day snapping, and whether a fixed reference clock exists.
- **Per-brand config** — tone, greeting, opt-out text, banned words, emoji policy — loaded by property and injected into the prompt and validators. RealPage serves thousands of properties; style can't be hardcoded.
- **Quiet-hours and TCPA rules** as a proper policy table (state-level hours, weekend rules, frequency caps).
- **LLM-judge as a second fair-housing check** on the draft, since lexical validators miss paraphrased violations.
- **Retrieval over approved messages** for few-shot, then **fine-tuning the generator** once there's volume. The `llm.complete()` boundary makes that a one-function swap.
- **Cadence execution** — each step calls this agent with the updated record and a `cadence_step`, so the same policy and generator produce every touch.
- **Observability** — decision reasons and validator outcomes as structured events, so a compliance team can audit why any message was or wasn't sent.
