# Operations Resolver Agent

Quest #4, Part 1 (Place-IL) — a single autonomous agent that resolves
GlobalCart support tickets: it reads the ticket, calls tools to look up the
order/customer/policy, decides on an operational outcome (auto-refund /
reject / escalate), and returns structured, auditable output.

> 🇮🇱 גרסה עברית: [`README.he.md`](README.he.md) — same content, same structure.

> 📓 [`demo.ipynb`](demo.ipynb) — a runnable walkthrough against the live
> API and real starter-kit tools (happy path, authority breach, hallucination
> trap, cross-customer authorization, the full 10-ticket regression suite,
> and the under-request-to-dodge-escalation guardrail), with real, saved
> outputs. This README is the design writeup; the notebook is the solution
> actually running.

The tool box and fixture data live in [`starter-kit/`](starter-kit/) and are
**unmodified**, exactly as provided — see [`starter-kit/README.md`](starter-kit/README.md)
for the tools themselves. Everything in this top-level README is the agent
built *around* that starter kit. The original assignment brief is kept for
reference in [`docs/quest-brief/`](docs/quest-brief/).

---

## Architecture

**A hand-rolled Anthropic tool-use loop, no agent framework.** The starter kit
is framework-agnostic by design — its own README ships ready-made adapters for
LangChain, CrewAI, PydanticAI and OpenAI Tools — so this was a choice, not a
constraint. Two reasons for it: the kit's `TOOL_SCHEMAS` are already in the
exact shape the Anthropic API expects (its README says so outright: *"TOOL_SCHEMAS
is already in the right shape"*), which removes the schema-translation layer a
framework mostly exists to provide; and what's left, the loop itself, is 202
lines of code (`resolver_agent/tool_loop.py`) and is exactly where every graded
guardrail lives — the forced final `submit_resolution` call, the repeat-call
refusal, the cross-customer denial at the dispatch boundary, and the typed
`ModelAPIError` that preserves a partial tool trace. A framework would hide that
loop rather than simplify it. The quest brief itself is neutral on the question
(see [`docs/quest-brief/agent-concepts-guide.md`](docs/quest-brief/agent-concepts-guide.md),
reading-list item #7).

```
resolver_agent/
├── tool_loop.py    generic send -> tool_use -> tool_result -> send engine
├── output_tool.py  the submit_resolution tool schema + a consistency validator
├── prompts.py      the system prompt
└── agent.py        ResolverAgent -- wires GlobalCart's 4 tools + submit_resolution
                     into tool_loop, exposes .resolve(ticket_text)
```

`tool_loop.py` is deliberately generic — it knows nothing about GlobalCart,
refunds, or JSON output shapes. It just runs the mechanical tool-calling
cycle for whatever `tool_schemas` / `tool_registry` it's handed. That's on
purpose: Part 2 turns this into a multi-agent team, and every agent in that
team can reuse this same loop unchanged, swapping in only a different prompt
and tool set.

`agent.py` is the only file that knows this is GlobalCart: it builds the tool
list (the 4 real tools from `mock_services.py` plus a 5th `submit_resolution`
tool, see below), supplies the system prompt, and turns the loop's raw
transcript into the three required output fields.

The model decides itself which of the 4 GlobalCart tools to call and in what
order — there is no hardcoded `order -> profile -> policy -> refund` pipeline
in the code. A typical case does end up calling all four in roughly that
order, but that's the model reasoning its way there via the tool
descriptions, not a fixed control-flow path.

### Prompt caching

The Anthropic API is stateless — every round-trip within one `resolve()` call
resends the full transcript, and the system prompt and 5-tool list are
identical on every one of those round-trips (neither changes mid-case).
`tool_loop.py` marks both with an Anthropic `cache_control` breakpoint
(`_cacheable_system` / `_cacheable_tools`), computed once per call and reused
for every round including the forced final one, so rounds after the first
reuse the cached prefix instead of the model reprocessing it from scratch.
Both helpers build new objects rather than mutating the caller's
`tool_schemas` in place — that list is built once in `ResolverAgent.__init__`
and reused across every `resolve()` call on the same instance, so mutating it
would leak a stale breakpoint (or worse, shared state) across unrelated
tickets.

### The tools: four from the starter kit, one of ours

The first four come from `starter-kit/mock_services.py` **unmodified**, via its
`TOOL_SCHEMAS` and `TOOL_REGISTRY`. They're listed in the order an agent
normally uses them — but again, that's the order the model tends to arrive at,
not an order written into the code.

| # | Tool | What it answers | Input | What comes back |
|---|---|---|---|---|
| 1 | `get_order_details` | What was ordered, when it arrived, what condition it arrived in | `order_id` | Shipping status, order/delivery dates, total amount, the items and the condition each arrived in, address and whether it changed after the order |
| 2 | `get_user_profile` | Who the customer is, their tier, refund history, risk | `user_id` | Tier (VIP → longer return window, higher cap), account age, LTV, refund history, prior fraud flags, fraud score |
| 3 | `check_return_policy` | Is this claim still eligible, and under which policy | `order_id`, `reason` | `eligible` + `verdict` (`ELIGIBLE` / `OUTSIDE_RETURN_WINDOW` / `NON_RETURNABLE_CATEGORY` / `ORDER_NOT_REFUNDABLE`), `applicable_policies`, `auto_refund_cap_usd`, `max_refundable_amount`, `requires_escalation` + `escalation_reasons`, `return_window_days` / `days_since_delivery` / `days_remaining_in_window`, `explanation` |
| 4 | `process_refund` | Issue the refund — or refuse and demand escalation | `order_id`, `amount`, `reason` | `status`: `APPROVED` / `REJECTED` / `ESCALATION_REQUIRED`, alongside `approved_amount`, `refund_id`, `requested_amount`, `auto_refund_cap_usd`, `applicable_policies`, `reasons`, `message` |
| 5 | `submit_resolution` | **Ours** — record the final resolution and end the case | `reasoning_chain`, `action_taken`, `customer_response` | No business result; it's the loop's "I'm done" signal (see below) |

Three facts about that contract drive the rest of the design:

| Fact | Consequence for the agent |
|---|---|
| `process_refund` is the **only tool with a side effect** | It's also the only one the final decision is cross-checked against, in `enforce_resolution` |
| The cap is enforced **inside the tool**, not in the prompt | A request above the cap returns `ESCALATION_REQUIRED`; no prompt talks it into `APPROVED` |
| A business failure is **data, not an exception** | `{"error": "ORDER_NOT_FOUND" / "USER_NOT_FOUND" / "INVALID_AMOUNT", "message": ...}` flows back to the model as an ordinary `tool_result` |

The tool *descriptions* were left untouched, and that matters: the quest's guide
says that if an agent picks the wrong tool, the fix belongs in the tool
description rather than the system prompt. The kit's descriptions are already
written that way — each says both *what* the tool does and *when* to call it
("Call this first for any ticket that mentions an order", "Call this before
promising the customer anything", "Call it only after check_return_policy
reported the claim eligible"). That's exactly why `prompts.py` is short: it
carries only what a tool description can't express — the agent's authority, and
the behavioral rules that keep it honest about what the tools actually said.

#### A real example: the tools in use

A table is a description; here's what actually happens. Every value below came
from calling the real starter-kit tools, not from memory. They're deterministic
— dates are computed against a fixed `reference_date() == 2026-08-05` (brief §5)
— and side-effect free, so all of it is reproducible with no API key.

> **What's fixed and what isn't:** the tool results below are exact and stable.
> The *sequence* is the run the model typically produces, not a guaranteed
> pipeline — as stated above, call order is the model's decision at runtime.

**🟢 Case 1 — the clean path (`ORD-1001`)**

Ticket: *"My earbuds from order ORD-1001 arrived cracked right out of the box."*

| # | Call | What came back |
|---|---|---|
| 1 | `get_order_details({order_id: "ORD-1001"})` | `user_id=USR-101`, `status=delivered`, `total_amount=35.0`, item condition `damaged_on_arrival` |
| 2 | `get_user_profile({user_id: "USR-101"})` | `tier=VIP`, `prior_fraud_flags=0`, `lifetime_value=4820.5` |
| 3 | `check_return_policy({order_id, reason: "damaged_on_arrival"})` | `eligible=true`, `verdict=ELIGIBLE`, `45`-day window (`11` elapsed, `34` left), `auto_refund_cap_usd=75.0`, `max_refundable_amount=35.0`, `applicable_policies=[POL-RET-02, POL-REF-02]` |
| 4 | `process_refund({order_id, amount: 35.0, reason})` | `status=APPROVED`, `approved_amount=35.0`, `refund_id=RF-1001-3500` |

Then `submit_resolution`, and what `resolve()` returns:

```json
{
  "reasoning_chain": [
    "ORD-1001 was delivered on 2026-07-25; one item (SKU-HDPH-01, 35.00 USD) is flagged damaged_on_arrival.",
    "USR-101 is a VIP customer with no prior fraud flags — 45-day window, 75.00 USD cap.",
    "check_return_policy returned ELIGIBLE: 11 days since delivery, 34 remaining (POL-RET-02, POL-REF-02).",
    "process_refund for 35.00 USD returned APPROVED with refund_id RF-1001-3500."
  ],
  "action_taken": {
    "tools_called": ["get_order_details", "get_user_profile", "check_return_policy", "process_refund"],
    "decision": "AUTO_REFUND_APPROVED",
    "refund_amount": 35.0,
    "refund_id": "RF-1001-3500"
  },
  "customer_response": "Hi Maya, sorry the earbuds turned up cracked...",
  "_case_id": "…", "_tool_calls": [ … ], "_validation_warnings": [], "_corrections": [], "_stopped_reason": "stop"
}
```

That's the test for `reasoning_chain`: every line is checkable against a row in
the table above — date, amount, policy id, `refund_id`. This is the difference
between an auditable chain and generic phrasing that would fit any ticket.

**🟡 Case 2 — authority breach (`ORD-1002`)**

The first three calls are the same shape. The divergence is the cap:

| Call | What came back |
|---|---|
| `get_order_details` | `total_amount=150.0` |
| `get_user_profile` | `tier=Standard` → a `50.0` cap, not `75.0` |
| `check_return_policy` | `eligible=true`, `verdict=ELIGIBLE`, `requires_escalation=false` — but `auto_refund_cap_usd=50.0` and `max_refundable_amount=50.0` |
| `process_refund({amount: 150.0})` | `status=ESCALATION_REQUIRED`, `approved_amount=0.0`, **no `refund_id`** |

The tool's own message: *"Refund not issued. 150.00 USD exceeds your automatic
authority of 50.00 USD — escalate to a human operations lead."*

Note the distinction: `check_return_policy` said `ELIGIBLE` with
`requires_escalation=false`. **Eligibility and authority are different
questions** — the claim is entirely legitimate, it's just above what the agent
may approve alone. An agent that reads `eligible=true` and skips the actual
`process_refund` result will promise a refund that never happened.

**🔴 Why that needs a guardrail in our code**

What if the agent asks for exactly the cap instead of the real amount? Checked
against the real tool:

```
process_refund({order_id: "ORD-1002", amount: 50.0})
  → status: APPROVED, approved_amount: 50.0, refund_id: "RF-1002-5000"
```

**`APPROVED`.** The tool can't tell an honest $50 claim from an agent shaving its
request to dodge escalation — its cap is enforced, but intent isn't. That's
exactly what `output_tool.py:251-281` is for: it spots `requested_amount == cap`
below the order total and overrides back to `ESCALATION_REQUIRED`. The sharpest
illustration that the kit's guardrail doesn't excuse us from having our own.

**⚫ Case 3 — the hallucination trap (`ORD-2222`)**

One call, and that's the run:

```
get_order_details({order_id: "ORD-2222"})
  → {"error": "ORDER_NOT_FOUND", "message": "No order found with id 'ORD-2222'."}
```

No further tool calls. No invented delivery date, no `process_refund` against an
order that doesn't exist. `prompts.py` rule 2 tells the agent to treat the
`error` key as a stop signal, and the decision comes out `CANNOT_RESOLVE` —
precisely why that fourth value exists.

### Forcing structured output: `submit_resolution` as a tool

Asking a model to free-type JSON at the end and parsing it with regex is
fragile — the quest's own guide calls this out explicitly as a common trap.
Instead, the required output shape (`reasoning_chain`, `action_taken`,
`customer_response`) is defined as a fifth tool, `submit_resolution`
(`resolver_agent/output_tool.py`). The model calls it as an ordinary
`tool_use` turn, so its arguments are already schema-validated by the API
before this code ever inspects them — no regex, no "hope it parses."

The model isn't forced to call it from turn one — it still has to decide on
its own which real tools to investigate with first. The system prompt tells
it to call `submit_resolution` last. As a safety net, if the loop is about to
hit `max_iterations` without the model calling it, one final turn is made
with `tool_choice` pinned to `submit_resolution`, so the agent always
terminates with valid structured output instead of trailing off mid-thought.

`action_taken.decision` has four values, not three — `AUTO_REFUND_APPROVED`,
`REJECTED`, `ESCALATION_REQUIRED`, and `CANNOT_RESOLVE`. The fourth exists
specifically for the hallucination-trap scenario: an order or user that
simply doesn't exist is neither an approval, a policy rejection, nor a
cap-based escalation, and forcing it into `REJECTED` would blur that
distinction in the output.

### Reasoning chain

`reasoning_chain` is a list of strings the model is instructed (in
`prompts.py`) to fill with concrete facts it actually saw in tool results —
order IDs, dollar amounts, dates, policy IDs — rather than generic phrasing
that could apply to any ticket. This is graded on whether it's *auditable*:
you should be able to check every line against the tool call log below it.

### Guarding against the decision/response gap

The single most important failure mode called out in the brief is an agent
that receives `ESCALATION_REQUIRED` from `process_refund` and still tells the
customer "your refund has been processed." Three independent layers guard
against this:

1. **Prompt-level**: `prompts.py` explicitly instructs the model to derive
   `decision` and `customer_response` from the actual last tool result, never
   from what it intended to happen.
2. **Schema-level**: `output_tool.validate_schema()` independently checks
   that a `submit_resolution` call actually has every required field and
   that `decision` is one of the four real values — not just trusting the
   Anthropic API's own tool-schema constraint. A structurally invalid call
   is treated exactly like the model never calling `submit_resolution` at
   all, and falls back to the same safe, tested escalation.
3. **Enforcement-level**: `output_tool.enforce_resolution()` cross-checks a
   (structurally valid) resolution's stated `decision` against the real
   `process_refund` result, and — this is the part that changed — **doesn't
   just flag a mismatch, it corrects it.** If the model claims
   `AUTO_REFUND_APPROVED` but `process_refund` actually returned
   `ESCALATION_REQUIRED`, the returned `decision`, `refund_amount`,
   `refund_id` and `customer_response` are deterministically overridden to
   match the tool's ground truth before `resolve()` ever returns them — no
   second LLM call, just a direct read of what the tool actually said. The
   original inconsistency stays visible in `_validation_warnings` (what was
   wrong) and `_corrections` (what was changed and why), so nothing is
   silently hidden — it's just no longer possible for the wrong message to
   be the one a caller actually receives. This mirrors the same "guardrail
   lives in code, not in a prompt" principle that `process_refund`'s own
   refund cap uses, applied one layer further out.

**`REJECTED` had no equivalent check.** A security review found this
three-layer guard was asymmetric: `AUTO_REFUND_APPROVED` is rigorously
cross-checked against `process_refund`'s real result, but a ticket that
talked the model into declaring `REJECTED` off invented/hallucinated policy
reasoning — without `check_return_policy` ever returning `eligible: False`,
or `process_refund` itself returning `REJECTED` — passed through with zero
warnings. `_find_issues()` now requires tool-level corroboration for
`REJECTED` the same way it already required it for approvals; an unbacked
`REJECTED` is escalated instead, same as any other decision/response-gap
violation.

### Requester identity and cross-customer authorization

By default `resolve(ticket_text)` will look up whatever `order_id`/`user_id`
the model finds in the ticket text, with no check on whether it belongs to
whoever actually submitted the ticket — appropriate for an internal
ops-console context, but a real gap for a customer-facing deployment. Passing
`resolve(ticket_text, requester_user_id="USR-101")` closes it: every
GlobalCart tool's successful result carries a `user_id` field naming the
owning customer (confirmed directly against `mock_services.py` — all four
tools include it, not just the two obvious lookups), so
`agent._authorize_tool_registry()` wraps all four and substitutes a
`{"error": "NOT_AUTHORIZED", ...}` dict whenever a result's owner doesn't
match the requester — **before** the real data ever reaches the model's
context, not just filtered out of the final customer-facing text. Because
`NOT_AUTHORIZED` uses the same `error`-key shape as every other business
failure in this codebase, it needs no special-casing anywhere else:
`prompts.py`'s existing "if a tool result contains an error key" rule and
`output_tool.py`'s existing "a tool errored, no refund was processed"
enforcement rule both already cover it for free. Omitting
`requester_user_id` (the default) reproduces today's unrestricted behavior
exactly, so no existing caller needed to change. Try it:

```bash
python3 run_ticket.py "My order ORD-1001 arrived damaged." USR-999   # USR-999 doesn't own ORD-1001 -- denied
python3 run_ticket.py "My order ORD-1001 arrived damaged." USR-101   # USR-101 does -- proceeds normally
```

**What this doesn't close:** `requester_user_id` is still just a string the
caller supplies — this package never verifies the caller actually *is* that
identity. That has to happen upstream (a session token, SSO, whatever this
agent sits behind); it's genuinely outside what code inside `resolver_agent/`
can do, since there's no identity provider here to check against.

What *is* closable in code is the failure mode where a customer-facing
deployment accidentally runs unrestricted because some call site forgot to
pass `requester_user_id` — that's what `ResolverAgent(client=...,
require_verified_requester=True)` guards against: `resolve()` then raises
immediately (before ever calling the model) if `requester_user_id` is
omitted, rather than silently degrading to "any record is fair game." It's a
fail-closed switch, not authentication — the default (`False`) preserves
today's unrestricted-when-omitted behavior exactly, appropriate for an
internal ops-console context.

### Triggering a real workflow for cases a human must act on

`submit_resolution` covers *respond* (`customer_response`) and *write* (the
returned dict, logged via `agent.case_resolved`/`agent.resolution_corrected`)
-- but until a case actually needs a human, nothing created an artifact one
could act on. A `customer_response` saying "this has been escalated" was only
ever a sentence in the reply, with no downstream effect.

`resolver_agent/escalation_workflow.py` closes that: any resolution whose
final `decision` is `ESCALATION_REQUIRED` or `CANNOT_RESOLVE` (the two
values that mean "a human still needs to look at this," after every
guardrail above has already run) gets a structural record appended to an
ops queue (`escalation_queue.jsonl` by default, override with
`ResolverAgent(escalation_queue_path=...)` or the `ESCALATION_QUEUE_PATH`
env var). `AUTO_REFUND_APPROVED`/`REJECTED` are terminal and trigger nothing.
The record deliberately excludes `customer_response` and the raw ticket text
-- same privacy stance as `logging_utils.py`, which never logs either for the
same reason (can contain a customer's name). This runs for every path that
can produce a final resolution -- the normal flow, the schema-invalid
fallback, and the API-failure fallback all funnel through
`ResolverAgent._finalize_workflow()` -- and the result is visible on the
returned dict as `_workflow_triggered`.

**Delivery beyond the local file:** set `ESCALATION_WEBHOOK_URL` and the
record is POSTed as JSON to that URL instead -- deliberately vendor-agnostic
rather than tied to one ticketing SDK, since Zendesk triggers, PagerDuty's
Events API, Opsgenie, Slack incoming webhooks, and a bespoke internal
endpoint are all "accepts a JSON POST." A failed delivery (network error,
timeout, non-2xx) never loses the record -- `build_webhook_writer()` catches
it, logs `escalation_workflow.webhook_delivery_failed`, and falls back to the
same local JSONL append the file-only default would have used. For a caller
that wants to wire in a real ticketing SDK directly instead of the generic
webhook path, `ResolverAgent(escalation_writer=...)` overrides the writer
entirely.

**Webhook egress hardening**, added after a security review of this feature:

- **HTTPS only.** `post_webhook()` refuses any URL whose scheme isn't
  `https` *before* attempting delivery (or even constructing the request) --
  an escalation record must never be attempted in cleartext. A refused URL
  falls back to the local file exactly like a network failure would.
- **`reasoning_chain` never leaves the process.** It's freeform LLM text
  (nothing about `prompts.py`'s reasoning-chain instructions guarantees it's
  PII-safe), so it's excluded from the webhook payload specifically --
  unlike the local queue file, a webhook target is an arbitrary,
  operator-configured third party. The full record (including
  `reasoning_chain`) is still what lands in the local fallback file if
  delivery fails, since that stays within the process's own trust boundary.
- **The URL itself is never logged with its query string.** Webhook auth is
  commonly a signed token in the query string (`?token=...`) -- a delivery
  failure logs and reports only `scheme://host/path`, never the full URL, so
  a credential embedded in the webhook URL can't end up in `stderr` output.

### Edge cases and guardrails

| Case | How it's handled |
|---|---|
| Refund request above the auto-approval cap | `process_refund` itself refuses and returns `ESCALATION_REQUIRED` — no prompt can talk it past the cap. The agent's job is only to recognize and report that honestly. |
| Return requested outside the window | `check_return_policy` returns `OUTSIDE_RETURN_WINDOW` citing `POL-RET-01`/`POL-RET-02`; the agent rejects and quotes the policy id. |
| Order or user that doesn't exist (hallucination trap) | Tools return `{"error": "ORDER_NOT_FOUND"/"USER_NOT_FOUND", ...}`. The system prompt instructs the agent to treat that key as a stop signal, not something to paper over with invented data — `decision` becomes `CANNOT_RESOLVE`. |
| Malformed input (negative amount, invalid reason, non-existent order passed to `process_refund`) | The tools return structured `{"error": ...}` dicts rather than raising; the agent reads the error and reports it instead of crashing or retrying blindly. |
| Repeating the same tool call | `tool_loop.py` tracks `(tool_name, args)` signatures already seen and refuses to re-execute an identical call, feeding back a message telling the model to stop retrying and act on what it already has. |
| Runaway loop | `max_iterations` (default 8) caps the number of tool-calling rounds; if hit, the loop forces a final `submit_resolution` call so the agent still returns a safe, structured answer (defaulting to escalation) instead of hanging. |
| API/network failure | The Anthropic SDK already retries connection errors and 408/409/429/5xx internally; `tool_loop.py` catches whatever reaches it afterward (retries exhausted, or an immediately non-retryable error like the 400 "credit balance too low" hit during live testing) and wraps it as `ModelAPIError`, preserving any partial tool trace. `ResolverAgent.resolve()` catches only that typed error and returns a safe `ESCALATION_REQUIRED` instead of crashing the caller — any other exception (a real bug) still propagates. |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

## Running it

**1. Sanity-check the (unmodified) starter kit first** — this only tests the
deterministic data/rule engine, not the agent:

```bash
python3 starter-kit/examples/verify_scenarios.py
# All 33 checks passed.
```

**2. Resolve a single ticket:**

```bash
python3 run_ticket.py "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. Can you sort this out?"
```

Prints the full structured JSON result to stdout (validation warnings, if
any, go to stderr).

**3. Run the full regression suite** — the agent against all 9 scenarios from
[`starter-kit/examples/scenarios.md`](starter-kit/examples/scenarios.md)
(scenarios 5 and 7 are each two orders in the brief, so this runs 10 tickets
covering all nine):

```bash
python3 run_scenarios.py
```

This is deliberately a *different* kind of check than
`starter-kit/examples/verify_scenarios.py`: that script tests whether the
data and rule engine are internally consistent (no LLM involved — it will
pass identically every time). `run_scenarios.py` tests whether the *agent*,
reading natural-language tickets and choosing its own tool calls, reaches the
same correct outcomes — which can vary run to run since it goes through a
live model call.

---

## Testing

There are three tiers of verification here, and they check different things:

| Suite | What it checks | Needs an API key? |
|---|---|---|
| `starter-kit/examples/verify_scenarios.py` | The data/rule engine (`mock_services.py`, untouched) is internally consistent | No |
| `pytest` (`tests/`) | `resolver_agent`'s own logic — loop mechanics, guardrails, output validation | No |
| `run_scenarios.py` | The *agent's* judgment end to end, against a live model | Yes |

The `pytest` suite covers the parts of the agent that don't require an LLM
call to verify: `tests/test_tool_loop.py` drives `run_tool_loop` with a
scripted fake "model" (see `tests/helpers.py`) so it can assert on the loop's
own mechanics — the repeat-call guard, both `max_iterations` termination
paths, unrecognized-tool handling, and TypeError wrapping — while every tool
call the fake model requests is still dispatched through the real
`starter-kit` functions, not a mock of them. `tests/test_output_tool.py`
does the same for `validate_resolution`, including a regression test for the
exact under-request-to-dodge-escalation bug caught during live testing.

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Logging

`resolver_agent` emits structured JSON logs (one object per line) to
**stderr** via the stdlib `logging` module — never stdout, which is reserved
for `run_ticket.py`'s actual result. Every log line from one `resolve()` call
carries the same `case_id`, so lines from concurrent or sequential cases
don't get tangled together. Level is controlled by `LOG_LEVEL` (default
`WARNING`, so a clean successful run prints nothing to stderr):

```bash
LOG_LEVEL=INFO python3 run_ticket.py "..."   2> >(jq .)   # pretty-print the logs separately
```

| Level | Event | When |
|---|---|---|
| `DEBUG` | `tool_loop.tool_executed` | every real GlobalCart tool call |
| `WARNING` | `tool_loop.repeat_call_refused` / `tool_loop.unknown_tool_requested` / `tool_loop.max_iterations_reached` | the loop's own guardrails firing |
| `WARNING` | `agent.resolution_corrected` | the stated decision was overridden to match what a tool actually returned |
| `WARNING` | `agent.fallback_resolution_used` | the model never called `submit_resolution`, or the call was structurally invalid |
| `ERROR` | `agent.api_error` | the Anthropic API call itself failed |
| `INFO` | `agent.case_resolved` | a case resolved cleanly, no warnings |

Never logged: the raw ticket text or the `customer_response` body — both can
contain the customer's name. Log fields stay structural: `case_id`,
`decision`, tool-call counts, `stopped_reason`, error type names.

`resolver_agent` never calls `logging.basicConfig()` itself — only
`run_ticket.py`/`run_scenarios.py` call `configure_logging()`, once, at
startup. Embedding the package elsewhere (e.g. Part 2) means calling that
yourself, or not, without it fighting over the root logger.
