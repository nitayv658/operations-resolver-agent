# Live-run evidence

A mentor review of Part 1 (see the feedback attached to this submission)
noted that everything about the agent's real, live tool-calling behavior was
backed only by documentation and by tests that *simulate* the model
(`ScriptedClient`) — the review environment had no `ANTHROPIC_API_KEY`, so
there was no saved proof of an actual run against the real Anthropic API.

This directory closes that gap: a real, unedited output from
`run_scenarios.py` against the live API, generated on 2026-08-27.

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
transcript). `run_scenarios.py`'s pass condition deliberately requires zero
validation warnings — it measures the model's raw judgment, not the
corrected final output — so this scenario counts as "not clean" even though
the customer never saw a wrong answer. This is the same behavior already
documented in the top-level README's guardrail section, now with a real
transcript proving it actually happens.

To reproduce:

```bash
python3 starter-kit/examples/verify_scenarios.py   # data/rule engine sanity check, no API key needed
LOG_LEVEL=INFO python3 run_scenarios.py > out.txt 2> logs.jsonl
```
