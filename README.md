# Operations Resolver Agent

Quest #4, Part 1 (Place-IL) — a single autonomous agent that resolves
GlobalCart support tickets: it reads the ticket, calls tools to look up the
order/customer/policy, decides on an operational outcome (auto-refund /
reject / escalate), and returns structured, auditable output.

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
cycle for whatever `tool_schemas` / `tool_registry` it's handed. `agent.py`
is the only file that knows this is GlobalCart: it builds the tool list, the
4 real tools from `mock_services.py` plus a 5th `submit_resolution` tool,
supplies the system prompt, and turns the loop's raw transcript into the
three required output fields.

The model decides itself which of the 4 GlobalCart tools to call and in what
order — there is no hardcoded `order -> profile -> policy -> refund`
pipeline in the code. It gets there by reasoning over the tool descriptions,
not a fixed control-flow path.

### Why a hand-rolled tool loop, not an agent framework

The starter kit is framework-agnostic by design — its own README ships
ready-made adapters for LangChain, CrewAI, PydanticAI and OpenAI Tools — so
this was a choice, not a constraint. Two reasons for it:

1. The kit's `TOOL_SCHEMAS` are already in the exact shape the Anthropic API
   expects (its README says so outright: *"TOOL_SCHEMAS is already in the
   right shape"*), which removes the schema-translation layer a framework
   mostly exists to provide.
2. What's left, the loop itself, is ~200 lines and is exactly where every
   graded guardrail lives — the forced final `submit_resolution` call, the
   repeat-call refusal, the cross-customer denial at the dispatch boundary,
   and the typed `ModelAPIError` that preserves a partial tool trace. A
   framework would hide that loop rather than simplify it.

It also pays off structurally: Part 2 turns this into a multi-agent team,
and every agent in that team can reuse this same loop unchanged, swapping in
only a different prompt and tool set.

### Prompt caching

The Anthropic API is stateless — every round-trip within one `resolve()`
call resends the full transcript, and the system prompt and 5-tool list are
identical on every one of those round-trips. `tool_loop.py` marks both with
an Anthropic `cache_control` breakpoint, computed once per call and reused
for every round including the forced final one, so rounds after the first
reuse the cached prefix instead of the model reprocessing it from scratch.

### The tools: four from the starter kit, one of ours

| # | Tool | What it answers |
|---|---|---|
| 1 | `get_order_details` | What was ordered, when it arrived, what condition it arrived in |
| 2 | `get_user_profile` | Who the customer is, their tier, refund history, risk |
| 3 | `check_return_policy` | Is this claim still eligible, and under which policy |
| 4 | `process_refund` | Issue the refund — or refuse and demand escalation |
| 5 | `submit_resolution` | **Ours** — record the final resolution and end the case |

Three facts about that contract drive the rest of the design:

| Fact | Consequence for the agent |
|---|---|
| `process_refund` is the **only tool with a side effect** | It's also the only one the final decision is cross-checked against |
| The refund cap is enforced **inside the tool**, not in the prompt | A request above the cap returns `ESCALATION_REQUIRED`; no prompt can talk it past the cap |
| A business failure is **data, not an exception** | `{"error": "ORDER_NOT_FOUND", ...}` flows back to the model as an ordinary `tool_result`, not a crash |

The tool *descriptions* were left untouched, and that matters: the quest's
guide says that if an agent picks the wrong tool, the fix belongs in the
tool description rather than the system prompt. The kit's descriptions
already say both *what* each tool does and *when* to call it, which is
exactly why `prompts.py` is short: it only carries what a tool description
can't express — the agent's authority, and the behavioral rules that keep it
honest about what the tools actually said.

### Forcing structured output: `submit_resolution` as a tool

Asking a model to free-type JSON at the end and parsing it with regex is
fragile — the quest's own guide calls this out explicitly as a common trap.
Instead, the required output shape (`reasoning_chain`, `action_taken`,
`customer_response`) is defined as a fifth tool (`resolver_agent/output_tool.py`).
The model calls it as an ordinary `tool_use` turn, so its arguments are
already schema-validated by the API before this code ever inspects them —
no regex, no "hope it parses."

The model isn't forced to call it from turn one — it still decides on its
own which real tools to investigate with first; the system prompt tells it
to call `submit_resolution` last. As a safety net, if the loop is about to
hit `max_iterations` without the model calling it, one final turn is made
with `tool_choice` pinned to `submit_resolution`, so the agent always
terminates with valid structured output instead of trailing off mid-thought.

`action_taken.decision` has four values, not three —
`AUTO_REFUND_APPROVED`, `REJECTED`, `ESCALATION_REQUIRED`, and
`CANNOT_RESOLVE`. The fourth exists specifically for the hallucination-trap
scenario: an order or user that simply doesn't exist is neither an
approval, a policy rejection, nor a cap-based escalation, and forcing it
into `REJECTED` would blur that distinction in the output.

### Guarding against the decision/response gap

The single most important failure mode called out in the brief is an agent
that receives `ESCALATION_REQUIRED` from `process_refund` and still tells
the customer "your refund has been processed." Three independent layers
guard against this:

1. **Prompt-level**: `prompts.py` explicitly instructs the model to derive
   `decision` and `customer_response` from the actual last tool result,
   never from what it intended to happen.
2. **Schema-level**: `output_tool.validate_schema()` independently checks
   that a `submit_resolution` call has every required field and a valid
   `decision` — not just trusting the API's own tool-schema constraint. A
   structurally invalid call falls back to the same safe, tested
   escalation as if `submit_resolution` had never been called at all.
3. **Enforcement-level**: `output_tool.enforce_resolution()` cross-checks
   the stated `decision` against the real `process_refund` result, and
   **doesn't just flag a mismatch, it corrects it** — deterministically
   overriding `decision`, `refund_amount`, `refund_id` and
   `customer_response` to match the tool's ground truth before `resolve()`
   ever returns, no second LLM call. The original inconsistency stays
   visible in `_validation_warnings`/`_corrections`, so nothing is silently
   hidden — it's just no longer possible for the wrong message to be the
   one a caller actually receives. This mirrors the same "guardrail lives
   in code, not in a prompt" principle `process_refund`'s own cap uses,
   applied one layer further out.

`REJECTED` gets the same treatment as `AUTO_REFUND_APPROVED`: a decision
declared off invented/hallucinated policy reasoning, without tool-level
corroboration, is escalated instead of passed through — a security review
found the original version of this check was asymmetric and only
cross-verified approvals.

**A real trap this catches:** if the agent asks `process_refund` for
exactly the cap amount instead of the customer's real (higher) claim, the
tool returns a clean `APPROVED` — it enforces its cap, but not intent. It
can't tell an honest claim at the cap from an agent shaving its request to
dodge escalation. `output_tool.py` spots `requested_amount == cap` below
the order total and overrides back to `ESCALATION_REQUIRED` — the sharpest
illustration that the kit's own guardrail doesn't excuse this code from
having its own.

### Requester identity and cross-customer authorization

By default `resolve(ticket_text)` looks up whatever `order_id`/`user_id` the
model finds in the ticket text, with no check on whether it belongs to
whoever actually submitted the ticket — appropriate for an internal
ops-console context, but a real gap for a customer-facing deployment.
Passing `resolve(ticket_text, requester_user_id="USR-101")` closes it: every
GlobalCart tool's successful result carries a `user_id` field naming the
owning customer, so `agent._authorize_tool_registry()` wraps all four tools
and substitutes a `{"error": "NOT_AUTHORIZED", ...}` dict whenever a
result's owner doesn't match the requester — **before** the real data ever
reaches the model's context, not just filtered out of the final
customer-facing text. `NOT_AUTHORIZED` uses the same `error`-key shape as
every other business failure in this codebase, so it needs no
special-casing anywhere else — `prompts.py`'s existing "error key means
stop" rule and `output_tool.py`'s existing "an errored tool means no refund
was processed" rule both already cover it for free. Omitting
`requester_user_id` (the default) reproduces today's unrestricted behavior
exactly, so no existing caller needed to change.

**What this doesn't close:** `requester_user_id` is still just a string the
caller supplies — this package never verifies the caller actually *is* that
identity. That has to happen upstream (a session token, SSO, whatever this
agent sits behind); it's genuinely outside what code inside
`resolver_agent/` can do without an identity provider to check against.
What *is* closable in code is the failure mode where a customer-facing
deployment accidentally runs unrestricted because some call site forgot to
pass `requester_user_id` — `ResolverAgent(require_verified_requester=True)`
makes `resolve()` raise immediately if it's omitted, rather than silently
degrading to "any record is fair game." It's a fail-closed switch, not
authentication — the default (`False`) preserves today's
unrestricted-when-omitted behavior exactly.

### Triggering a real workflow for cases a human must act on

`submit_resolution` covers *respond* and *write*, but until a case actually
needs a human, nothing created an artifact one could act on — a
`customer_response` saying "this has been escalated" was only ever a
sentence in the reply.

`resolver_agent/escalation_workflow.py` closes that: any resolution whose
final `decision` is `ESCALATION_REQUIRED` or `CANNOT_RESOLVE` gets a
structural record appended to an ops queue (a local JSONL file by default).
`AUTO_REFUND_APPROVED`/`REJECTED` are terminal and trigger nothing. The
record deliberately excludes `customer_response` and the raw ticket text —
same privacy stance the logging takes, since either can carry a customer's
name.

**Delivery beyond the local file:** setting `ESCALATION_WEBHOOK_URL` POSTs
the record as JSON to that URL instead — deliberately vendor-agnostic
rather than tied to one ticketing SDK, since Zendesk triggers, PagerDuty's
Events API, Opsgenie, Slack incoming webhooks, and a bespoke internal
endpoint are all "accepts a JSON POST." A failed delivery never loses the
record; it falls back to the same local JSONL append the file-only default
uses. A caller that wants a real ticketing SDK instead of the generic
webhook path can override the writer entirely.

**Webhook egress hardening**, added after a security review of this
feature:

- **HTTPS only** — a non-`https` webhook URL is refused before the request
  is even constructed; an escalation record must never go out in cleartext.
- **`reasoning_chain` never leaves the process** — it's freeform LLM text
  with no guarantee it's PII-safe, so it's excluded from the webhook
  payload specifically, unlike the local queue file which stays inside the
  process's own trust boundary.
- **The URL is never logged with its query string** — webhook auth is
  commonly a token in the query string, so failures log only
  `scheme://host/path`, never the full URL.

### Other guardrails

| Case | How it's handled |
|---|---|
| Order or user that doesn't exist (hallucination trap) | Tools return `{"error": ...}`; the prompt treats that as a stop signal, not something to paper over — `decision` becomes `CANNOT_RESOLVE`. |
| Repeating the same tool call | `tool_loop.py` tracks `(tool_name, args)` signatures already seen and refuses to re-execute, telling the model to act on what it already has. |
| Runaway loop | `max_iterations` (default 8) caps the tool-calling rounds; if hit, the loop forces a final `submit_resolution` call instead of hanging or trailing off. |
| API/network failure | The SDK already retries connection errors and 408/409/429/5xx; anything that still reaches `tool_loop.py` is wrapped as a typed `ModelAPIError`, and `resolve()` turns that into a safe `ESCALATION_REQUIRED` rather than crashing the caller. Any other exception (a real bug) still propagates. |

---

## Testing this design

There are three intentionally different tiers of verification:

| Suite | What it checks | Needs an API key? |
|---|---|---|
| `starter-kit/examples/verify_scenarios.py` | The data/rule engine (`mock_services.py`, untouched) is internally consistent | No |
| `pytest` (`tests/`) | `resolver_agent`'s own logic — loop mechanics, guardrails, output validation — driven with a scripted fake model, dispatched through the real starter-kit tools | No |
| `run_scenarios.py` | The *agent's* judgment end to end, against a live model, across all 9 brief scenarios | Yes |

The split matters: the first tests whether the fixtures and rules are
consistent, the second tests this package's own code paths deterministically,
and only the third tests whether the model actually reasons its way to the
right outcome — which is the one that can vary run to run.

**A real `run_scenarios.py` transcript is saved in [`docs/evidence/`](docs/evidence/)** —
stdout and structured logs from an actual run against the live API, not just
this README's description of one, so the third tier's claims hold up even
without your own API key.

---

## Part 2 — Distributed Agent Crew

> 📄 [`docs/evidence/`](docs/evidence/) — a real, saved transcript of
> `run_crew_scenarios.py` against the live API (stdout + structured JSON
> logs, 6/6 scenarios matched), so the crew's real tool-calling behavior
> below is backed by an actual run, not just this section's description of
> one. (There's no `demo.ipynb`-style notebook for the crew yet — this is
> the Part 2 equivalent for now.)

Part 2 turns the single `ResolverAgent` into a three-agent pipeline —
**Researcher → Decision → Comms** — each with a narrower job, a narrower
tool bundle, and its own prompt. `resolver_agent/tool_loop.py` is reused
completely unchanged: it never knew about GlobalCart specifically, so the
same send → tool_use → tool_result → send engine that ran one agent in Part
1 runs each of these three, just handed a different `tool_schemas` /
`tool_registry` / prompt each time.

```
resolver_agent/crew/
├── schemas.py        RiskReport / Decision / CrewResult -- the handoff contracts
├── orchestrator.py   OperationsCrew.handle_ticket -- wires Researcher -> Decision -> Comms
├── researcher/        Agent 1 -- get_order_details, get_user_profile, audit_fraud_risk
├── decision/           Agent 2 -- check_return_policy, process_refund
└── comms/               Agent 3 -- get_escalation_route, send_slack_alert
```

Each of the three agent packages mirrors `resolver_agent/`'s own Part 1
shape one-to-one — an `agent.py`, an `output_tool.py`
(`validate_schema` + `enforce_*`), and a `prompts.py` — because the pattern
that shape encodes (force the handoff as a tool call, then cross-check it
against the real tool result rather than trust it) is exactly what each
stage needs too, just with a different bundle of tools and a different
schema at the end of it.

### End-to-end flow

Every arrow between stages passes through `OperationsCrew.handle_ticket` —
no agent calls another, and none of them share a conversation. Each stage
opens a fresh message list holding only the upstream object serialized as
JSON, so the model downstream reads real structured data rather than a
paraphrase of what the previous model said.

```
                         run_crew.py / run_crew_scenarios.py
                                       │  ticket_text
                                       ▼
              ┌─────────────────────────────────────────────────┐
              │  OperationsCrew.handle_ticket   (orchestrator)  │
              │  case_id = uuid4[:8]  ──> every log line        │
              └─────────────────────────────────────────────────┘
                                       │
   ticket_text (raw str) ──────────────┘
                                       ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ AGENT 1  RESEARCHER          system: RESEARCHER_PROMPT                ║
║ tools: get_order_details · get_user_profile · audit_fraud_risk        ║
║ stop tool: submit_risk_report                                         ║
║   _extract_report ─> validate_schema ─> enforce_risk_report           ║
║        (drift vs. real audit_fraud_risk result is overwritten)        ║
╚═══════════════════════════════════════════════════════════════════════╝
                                       │  ResearcherResult
                     ┌─────────────────┴─────────────────┐
              report is None                      report is RiskReport
                     │                                   │
                     ▼                                   │
        ┌────────────────────────────┐                   │
        │ STOP: researcher_incomplete│                   │
        │ generic reply, no retry    │                   │
        └────────────────────────────┘                   │
                                                         │
        risk_report.model_dump_json()  ──────────────────┘
                                       ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ AGENT 2  DECISION            system: DECISION_PROMPT                  ║
║ tools: check_return_policy · process_refund      <- only money holder ║
║ stop tool: submit_decision                                            ║
║   _extract_decision ─> validate_schema ─> enforce_decision            ║
║        (refund_status cross-checked vs. process_refund + risk report) ║
║   builds Decision, nesting the FULL RiskReport                        ║
╚═══════════════════════════════════════════════════════════════════════╝
                                       │  DecisionResult
                     ┌─────────────────┴─────────────────┐
             decision is None                     decision is Decision
                     │                                   │
                     ▼                                   │
        ┌────────────────────────────┐                   │
        │ STOP: decision_incomplete  │                   │
        │ escalation reply, no retry │                   │
        └────────────────────────────┘                   │
                                                         │
        decision.model_dump_json()  ─────────────────────┘
                                       ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ AGENT 3  COMMS               system: COMMS_PROMPT                     ║
║ tools: get_escalation_route · send_slack_alert   <- no refund access  ║
║ _guarded_registry: alert refused unless THIS case's route said        ║
║                    escalation_required=true (fresh state per run)     ║
║ stop tool: submit_comms_result  ──> customer_response only            ║
║ escalation / alert_sent / alert_record read from real tool_calls      ║
║ no submit? ──> canned reply keyed on refund_status, case still ends   ║
╚═══════════════════════════════════════════════════════════════════════╝
                                       │  CommsResult
                                       ▼
              ┌─────────────────────────────────────────────────┐
              │ CrewResult                                      │
              │ order_id · customer_response · decision         │
              │ escalation · alert_sent · alert_record          │
              │ reasoning_chain (all 3 stages) · stopped_reason │
              └─────────────────────────────────────────────────┘
```

Two asymmetries are worth reading off the diagram. The Researcher and
Decision stages each have a hard stop — a missing report or a missing
decision ends the case with a safe reply, and the failing agent is never
re-dispatched, since its own `max_iterations` already bounds a single
stage's runaway. The Comms stage has no such stop: only its
`customer_response` comes from the model, while `escalation`, `alert_sent`,
and `alert_record` are read back out of the real `tool_calls` records, so a
missing `submit_comms_result` degrades to a canned reply instead of failing
the case.

### Why three agents, not one prompt with more tools

Part 1's `submit_resolution` already showed that a single agent juggling
four tools and a refund cap works. The Stage 2 brief adds fraud auditing
and Slack alerting on top of that, and the brief's own central trap —
`ORD-1005`, a claim `check_return_policy` calls **`ELIGIBLE`** on its own,
that only `audit_fraud_risk`'s 90/100 high-risk score actually blocks — is
a fraud signal a policy-only tool has no way to see. One agent could still
technically hold all seven tools, but splitting by *responsibility* buys
something a bigger single prompt can't: financial authority and outbound
messaging become **physically** separated (next section), not just
separated by an instruction the model could be talked out of.

### Authority separation is physical, not just prompted

Each agent's `self.tool_registry` is built from a fixed tuple of tool
names, one per role — matching the starter kit's own
`multi_agent_tools.TOOL_OWNERSHIP` map:

| Agent | Can call | Cannot do |
|---|---|---|
| **Researcher** | `get_order_details`, `get_user_profile`, `audit_fraud_risk` | Cannot approve/reject a refund or talk to the customer |
| **Decision** | `check_return_policy`, `process_refund` | Cannot look anything up itself; cannot talk to the customer |
| **Comms** | `get_escalation_route`, `send_slack_alert` | Cannot touch money |

`DecisionAgent` is the only one of the three whose `tool_registry` can even
reach `process_refund` — `ResearcherAgent`'s and `CommsAgent`'s registries
don't contain it at all, so there's no code path by which either could call
it even if a prompt injection or a reasoning error told them to. This is
the same "the guardrail lives in how the registry is built, not in a
sentence the model reads" principle Part 1's `_authorize_tool_registry`
guardrail uses, applied to role separation instead of customer
authorization.

### The handoff contracts carry the full report, not a summary

`schemas.py`'s `Decision` model embeds the entire upstream `RiskReport`,
not just its `risk_band` string. That's a deliberate choice forced by
`ORD-1005`: if the Researcher-to-Decision handoff only carried a `"high"`
label, the Decision agent would have to trust that label instead of being
able to see *why* — the `risk_score`, the `triggered_rules`, the raw
`evidence` — which is exactly what it needs to weigh against a policy tool
that, on this same case, is saying `ELIGIBLE`. Losing that detail at the
handoff boundary would silently defeat the point of the Researcher's work
one stage later.

### Guardrails specific to the crew

| Case | How it's handled |
|---|---|
| Researcher can't find the order/user, or the user doesn't own the order (`USER_ORDER_MISMATCH`) | `ResearcherAgent.run()` returns `report=None`; `OperationsCrew` stops the pipeline immediately and returns a safe, generic escalation response — it never re-dispatches the Researcher, since a retry loop is the wrong response to missing data |
| Decision agent never produces a valid `submit_decision` call | Same stop-and-escalate treatment, one stage later, with the `RiskReport` still included in the trace |
| `send_slack_alert` called without *this same case's* own `get_escalation_route` call having returned `escalation_required: true` | `comms/agent.py`'s `_guarded_registry` denies the call in code with `ALERT_NOT_AUTHORIZED` — built fresh per `run()` call, so one case's routing result can never leak into another case's alert decision |
| A stage's own model output looks structurally valid but doesn't match what the real tool actually returned | `enforce_risk_report` / `enforce_decision` cross-check and deterministically correct it — the same "guardrail lives in code, not in a prompt" pattern as Part 1's `output_tool.enforce_resolution`, run at every stage instead of once |
| `CommsAgent`'s own `customer_response` is missing or structurally invalid | Falls back to a safe, generic reply derived from `decision.refund_status` (mirrors Part 1's `_safe_customer_response`) |

### Running it

```
python3 run_crew.py "This is Ronen, order ORD-1005. The tablet screen ..."
```

prints the resulting `CrewResult` as JSON (requires `ANTHROPIC_API_KEY`; set
`SLACK_WEBHOOK_URL` to also POST any alert to a real Slack incoming
webhook, otherwise it's written to `starter-kit/outbox/alerts.jsonl`).
`run_crew_scenarios.py` runs the crew against the Stage 2 brief's headline
scenarios plus a Part 1 regression spot-check, the same role
`run_scenarios.py` plays for Part 1.

### Testing this design

`tests/crew/` (26 tests) follows the same split Part 1 uses: a scripted
fake model drives each agent and the orchestrator through the real
starter-kit tools, so the whole suite — 136 tests total across both parts —
runs deterministically with no API key. `tests/crew/test_tool_ownership.py`
specifically asserts each agent's registry only contains the tool names
`TOOL_OWNERSHIP` assigns it, so the authority-separation guarantee above is
checked in CI, not just true by construction today.

**A real `run_crew_scenarios.py` transcript is saved in
[`docs/evidence/`](docs/evidence/)** — stdout and structured logs from an
actual run against the live API, not just this README's description of one,
so the crew's real tool-calling behavior and its corrections (`decision.corrected`
firing for real on the `P1-2` case) hold up even without your own API key.
