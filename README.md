# Operations Resolver Agent

Quest #4, Part 1 (Place-IL) — a single autonomous agent that resolves
GlobalCart support tickets: it reads the ticket, calls tools to look up the
order/customer/policy, decides on an operational outcome (auto-refund /
reject / escalate), and returns structured, auditable output.

The tool box and fixture data live in [`starter-kit/`](starter-kit/) and are
**unmodified**, exactly as provided — see [`starter-kit/README.md`](starter-kit/README.md)
for the tools themselves. Everything in this top-level README is the agent
built *around* that starter kit. The original assignment brief is kept for
reference in [`docs/quest-brief/`](docs/quest-brief/).

---

## Architecture

**A hand-rolled Anthropic tool-use loop, no agent framework.** The starter
kit's `TOOL_SCHEMAS` are already in the exact shape the Anthropic API expects,
and the loop itself is short enough (`resolver_agent/tool_loop.py`) that a
framework would mostly be hiding it rather than simplifying it — which is
also the quest's own recommendation (see
[`docs/quest-brief/agent-concepts-guide.md`](docs/quest-brief/agent-concepts-guide.md)).

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
| `WARNING` | `agent.validation_warnings` | the stated decision didn't match what a tool actually returned |
| `WARNING` | `agent.fallback_resolution_used` | the model never called `submit_resolution` |
| `ERROR` | `agent.api_error` | the Anthropic API call itself failed |
| `INFO` | `agent.case_resolved` | a case resolved cleanly, no warnings |

Never logged: the raw ticket text or the `customer_response` body — both can
contain the customer's name. Log fields stay structural: `case_id`,
`decision`, tool-call counts, `stopped_reason`, error type names.

`resolver_agent` never calls `logging.basicConfig()` itself — only
`run_ticket.py`/`run_scenarios.py` call `configure_logging()`, once, at
startup. Embedding the package elsewhere (e.g. Part 2) means calling that
yourself, or not, without it fighting over the root logger.

---

## Demo video

*(optional, ≤2 min — add a link here showing a happy-path run and an
edge-case run via `run_ticket.py` or `run_scenarios.py`)*

---

## A note on the fixtures

The brief asks us to flag rather than edit anything that looks off in
`starter-kit/data/`. Nothing was found that needed flagging — all 33 checks
in `verify_scenarios.py` pass against the fixtures as provided.
