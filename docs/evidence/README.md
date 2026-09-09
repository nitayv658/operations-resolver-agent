# Live-run evidence

Real, unedited transcripts from running this repo's scenario scripts against
the live Anthropic API — not just a README's description of what the agent
does, but proof it actually happened, generated on a specific date and
reviewable without needing your own API key.

## Part 1 — Single-Agent Resolver (`scripts/run_scenarios.py`)

A mentor review of Part 1 (see the feedback attached to this submission)
noted that everything about the agent's real, live tool-calling behavior was
backed only by documentation and by tests that *simulate* the model
(`ScriptedClient`) — the review environment had no `ANTHROPIC_API_KEY`, so
there was no saved proof of an actual run against the real Anthropic API.

This directory closes that gap: a real, unedited output from
`scripts/run_scenarios.py` against the live API, generated on 2026-08-27.

- [`run_scenarios_live_output.txt`](run_scenarios_live_output.txt) — stdout:
  all 10 tickets (covering the 9 brief scenarios), each with the tools the
  agent actually called, the decision it reached, and the customer-facing
  reply it wrote.
- [`run_scenarios_live_logs.jsonl`](run_scenarios_live_logs.jsonl) — stderr:
  the structured JSON log lines (`agent.case_resolved`,
  `agent.workflow_triggered`, ...) emitted for the same run, one object per
  line, correlated by `case_id` with the transcript above.

**Result: 9/10 scenarios matched the expected decision cleanly.** The one
exception (Scenario 5b, `ORD-1011` at $52 against the $50 cap) is not a bug —
it's the under-request-to-dodge-escalation guardrail firing for real: the
model requested exactly $50 (the cap) instead of the true $52 owed, and
`enforce_resolution` caught and corrected it to `ESCALATION_REQUIRED` before
it ever reached the customer (see the `validation warnings` line in the
transcript). `scripts/run_scenarios.py`'s pass condition deliberately requires zero
validation warnings — it measures the model's raw judgment, not the
corrected final output — so this scenario counts as "not clean" even though
the customer never saw a wrong answer. This is the same behavior already
documented in the top-level README's guardrail section, now with a real
transcript proving it actually happens.

## Part 2 — Distributed Agent Crew (`scripts/run_crew_scenarios.py`)

- [`run_crew_scenarios_live_output.txt`](run_crew_scenarios_live_output.txt) —
  stdout: all 6 scenarios (the Stage 2 headline fraud/clean cases plus a
  Part 1 regression spot-check), each with the `refund_status` the crew
  reached and the customer-facing reply the Comms agent wrote.
- [`run_crew_scenarios_live_logs.jsonl`](run_crew_scenarios_live_logs.jsonl) —
  stderr: the structured JSON log lines emitted by all three agents and the
  orchestrator for the same run (`researcher.report_produced`,
  `decision.decision_produced`, `decision.corrected`, `comms.result_produced`,
  `crew.case_resolved`, ...), one object per line, correlated by `case_id`
  with the transcript above.

**Result: 6/6 scenarios matched, generated 2026-09-09.** Pass criteria
(exactly what `scripts/run_crew_scenarios.py` checks): the crew's final
`refund_status` equals the scenario's expected value, *and* an alert lands
in `starter-kit/outbox/alerts.jsonl` exactly when — and only when —
expected.

### Each scenario, expected vs. actual

| # | Scenario | Ticket | Expected | Actual |
|---|---|---|---|---|
| B1 | Headline fraud trap — policy says `ELIGIBLE`, fraud engine says risk 90/100 | *"This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me the full 480 dollars, this keeps happening."* | `ESCALATION_REQUIRED` + alert to `#fraud-security` | ✅ Matched. `check_return_policy` returns `ELIGIBLE`; `audit_fraud_risk` independently flags `risk_band=high` (score 90, 5 fraud rules triggered: repeat claims, last-minute address change, prior fraud flag, high fraud score, refund velocity). The Decision agent correctly treats the fraud finding as a hard block over the clean policy verdict, never calls `process_refund`, and reports the real `$480.00` claim. |
| B2 | New account, high-value item never arrived | *"I ordered a laptop (ORD-1012), the box arrived but it was empty. I need the 890 dollars back."* | `ESCALATION_REQUIRED` + alert to `#fraud-security` | ✅ Matched. `risk_band=high` (score 60: address changed 1 day before delivery, 8-day-old account on an $890 order, high-value + item-never-arrived claims). Same hard-block pattern as B1; correctly reports the real `$890.00` claim. |
| B3 | Clean case — no fraud signal | *"Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. Can you sort this out?"* | `APPROVED`, no alert | ✅ Matched. `risk_band=low` (score 0), policy eligible within the automatic refund cap. `process_refund` approves the full `$35.00` (refund `RF-1001-3500`); no escalation, no alert — the crew doesn't page anyone on a case that resolved cleanly. |
| P1-2 | Part 1 regression — authority breach still escalates | *"Order ORD-1002. The espresso machine is dented and leaking. I paid 150 dollars for this. I want my money back today."* | `ESCALATION_REQUIRED` + alert to `#support-tier2` | ✅ Matched, **with the guardrail firing for real** — see below. |
| P1-3 | Part 1 regression — outside the return window still rejects | *"I ordered a backpack back at the end of May (ORD-1003) and I've changed my mind, I'd like to return it."* | `REJECTED` + alert to `#support-tier2` | ✅ Matched. `check_return_policy` returns `verdict=OUTSIDE_RETURN_WINDOW` (delivered 60 days ago, 30-day window) — rejected on policy grounds before fraud risk is even relevant. Still routed to Tier 2 for human review of the reply, not paged as fraud. |
| P1-8 | Part 1 regression — non-returnable category still rejects | *"ORD-1008, I bought a gift card by accident. Please refund it."* | `REJECTED` + alert to `#support-tier2` | ✅ Matched, **with a smaller correction firing** — see below. |

### The two guardrails caught firing for real in this run

Both are documented as design decisions in the top-level [README](../../README.md#part-2--distributed-agent-crew); this run shows each actually firing, not just being reasoned about — visible in the JSONL, not the stdout transcript, since `scripts/run_crew_scenarios.py` only prints the final `refund_status` and reply.

- **P1-2** (`case_id=18177edb`): the Decision agent called `process_refund`
  for exactly `$50.00` — the auto-refund cap — instead of the real `$150.00`
  order total, and `process_refund` came back a clean `APPROVED` (it
  enforces its cap, not intent). `decision.corrected` fires with
  `correction_count=2`: `resolver_agent/crew/decision/output_tool.py`'s
  cap-leak check caught `requested_amount == cap < order_total`, overrode
  `refund_status` from `APPROVED` to `ESCALATION_REQUIRED`, and corrected
  `requested_amount` back to the real `$150.00` — deterministically, no
  second model call. The customer never saw a wrong answer.
- **P1-8** (`case_id=47016fef`): the Researcher's own `action_hint` text
  drifted from what `audit_fraud_risk` actually returned (it added an
  unprompted note about verifying digital-goods refund policy).
  `researcher.report_corrected` fires with `correction_count=1`:
  `enforce_risk_report` overwrote it back to match the tool's real output
  before it reached the Decision agent.

To reproduce:

```bash
python3 starter-kit/examples/verify_scenarios.py   # data/rule engine sanity check, no API key needed
pytest tests/                                       # this package's own logic, scripted, no API key needed
LOG_LEVEL=INFO python3 scripts/run_scenarios.py > out.txt 2> logs.jsonl        # Part 1, needs ANTHROPIC_API_KEY
LOG_LEVEL=INFO python3 scripts/run_crew_scenarios.py > out.txt 2> logs.jsonl   # Part 2, needs ANTHROPIC_API_KEY
```
