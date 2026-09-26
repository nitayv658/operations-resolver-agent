"""Tests for resolver_agent.crew.jev_gate.run_gate -- the Jev-scored triage
gate inserted between Researcher and Decision.

Only the model is faked (ScriptedClient); check_return_policy/process_refund
are dispatched through the real starter-kit multi_agent_tools, same
discipline as tests/crew/decision/test_agent.py. MockJevClient's `scorer` is
used to force a specific branch deterministically -- these tests are about
the gate's own logic, not about Jev's (unverified, mocked) scoring itself.
"""

from __future__ import annotations

from resolver_agent.crew import jev_gate
from resolver_agent.crew.decision.output_tool import SUBMIT_DECISION_TOOL_NAME
from resolver_agent.crew.jev_gate import TRIAGE_QUERY_TOOL_NAME, run_gate
from resolver_agent.crew.schemas import RiskReport
from resolver_agent.jev_client import JevScore, MockJevClient

from ..helpers import ScriptedClient, ScriptedResponse, text_block, tool_use_block


def _risk_report(**overrides) -> RiskReport:
    base = dict(
        order_id="ORD-1001",
        user_id="USR-101",
        risk_score=0,
        risk_band="low",
        action_hint="proceed with the normal refund flow",
        triggered_rules=[],
        evidence={"order_total_usd": 35.0, "order_status": "delivered", "prior_fraud_flags": 0},
        blocks_automatic_refund=False,
        requires_security_channel=False,
        rulebook_version="1.0.0",
    )
    base.update(overrides)
    return RiskReport(**base)


def _extraction_response(**overrides):
    payload = {
        "reason": "damaged_on_arrival",
        "claim_summary": "Customer says the item arrived damaged.",
        "customer_stated_amount": None,
        "sentiment": "calm",
        "urgency": "low",
    }
    payload.update(overrides)
    return ScriptedResponse([tool_use_block(TRIAGE_QUERY_TOOL_NAME, payload)])


def _submit_response(**overrides):
    payload = {
        "order_id": "ORD-1001",
        "user_id": "USR-101",
        "verdict": "ELIGIBLE",
        "eligible": True,
        "refund_status": "APPROVED",
        "requested_amount": 35.0,
        "approved_amount": 35.0,
        "refund_id": "RF-1001-3500",
        "applicable_policies": ["POL-RET-02"],
        "rationale": "Eligible and within the auto-refund cap.",
    }
    payload.update(overrides)
    return ScriptedResponse([tool_use_block(SUBMIT_DECISION_TOOL_NAME, payload)])


def _forced_scorer(score: float, confidence: float):
    return MockJevClient(scorer=lambda query: JevScore(score=score, confidence=confidence, raw={"forced": True}))


def test_run_gate_fast_approve_calls_process_refund_for_real(mat_module):
    client = ScriptedClient([_extraction_response(), _submit_response()])

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="My ORD-1001 earbuds arrived cracked, please refund me.",
        risk_report=_risk_report(),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_forced_scorer(score=0.99, confidence=0.99),
        case_id="c1",
    )

    assert outcome.branch == "fast_approve"
    assert outcome.decision_result.error is None
    assert outcome.decision_result.decision.refund_status == "APPROVED"
    assert outcome.decision_result.decision.approved_amount == 35.0
    refund_calls = [c for c in outcome.decision_result.tool_calls if c.name == "process_refund"]
    assert len(refund_calls) == 1
    assert refund_calls[0].result["status"] == "APPROVED"
    # Only 2 model calls: extraction + the forced write-up -- no freeform loop.
    assert client.calls == 2


def test_run_gate_fast_reject_never_calls_process_refund(mat_module):
    # ORD-1003 is genuinely outside the return window (eligible=False) --
    # real fixture data, not a fake.
    client = ScriptedClient(
        [
            _extraction_response(),
            _submit_response(
                order_id="ORD-1003",
                user_id="USR-103",
                verdict="OUTSIDE_RETURN_WINDOW",
                eligible=False,
                refund_status="REJECTED",
                requested_amount=42.5,
                approved_amount=None,
                refund_id=None,
                applicable_policies=["POL-RET-01"],
                rationale="Outside the return window.",
            ),
        ]
    )

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="My ORD-1003 order arrived late, I want a refund.",
        risk_report=_risk_report(order_id="ORD-1003", user_id="USR-103", evidence={"order_total_usd": 42.5, "order_status": "delivered", "prior_fraud_flags": 0}),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_forced_scorer(score=0.01, confidence=0.99),
        case_id="c2",
    )

    assert outcome.branch == "fast_reject"
    assert outcome.decision_result.decision.refund_status == "REJECTED"
    refund_calls = [c for c in outcome.decision_result.tool_calls if c.name == "process_refund"]
    assert refund_calls == []
    assert client.calls == 2


def test_run_gate_low_confidence_falls_through_seeded_with_grounding(mat_module):
    client = ScriptedClient([_extraction_response()])

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="My ORD-1001 earbuds arrived cracked, please refund me.",
        risk_report=_risk_report(),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_forced_scorer(score=0.9, confidence=0.2),  # confident enough score, NOT confident enough overall
        case_id="c3",
    )

    assert outcome.branch == "fallthrough"
    assert outcome.decision_result is None
    assert outcome.seed_messages is not None
    # The gate's own check_return_policy call is real grounding, present in the seed.
    assert any(
        isinstance(m.get("content"), list) and any(b.get("name") == "check_return_policy" for b in m["content"] if isinstance(b, dict))
        for m in outcome.seed_messages
    )
    assert outcome.seed_seen_calls
    assert client.calls == 1  # extraction only -- no forced write-up call was made


def test_run_gate_ambiguous_score_falls_through(mat_module):
    client = ScriptedClient([_extraction_response()])

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="My ORD-1001 earbuds arrived cracked, please refund me.",
        risk_report=_risk_report(),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_forced_scorer(score=0.5, confidence=0.95),
        case_id="c4",
    )

    assert outcome.branch == "fallthrough"


def test_run_gate_blocks_automatic_refund_prevents_fast_approve_even_with_a_confident_score(mat_module):
    # The ORD-1005-style guardrail: even a maximally confident, high Jev
    # score must never reach fast_approve on a fraud-blocked report -- the
    # gate itself must not attempt process_refund here, not merely rely on
    # enforce_decision to correct it after the fact.
    client = ScriptedClient([_extraction_response()])

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="My ORD-1001 earbuds arrived cracked, please refund me.",
        risk_report=_risk_report(blocks_automatic_refund=True, risk_band="high", risk_score=90),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_forced_scorer(score=1.0, confidence=1.0),
        case_id="c5",
    )

    assert outcome.branch == "fallthrough"


def test_run_gate_extraction_failure_falls_through_with_no_grounding(mat_module):
    # The model never calls extract_triage_query at all -- including on
    # tool_loop's own forced retry (stop_tool_name pinned), which also stalls.
    client = ScriptedClient(
        [
            ScriptedResponse([text_block("I'm not sure.")], stop_reason="end_turn"),
            ScriptedResponse([text_block("Still not sure.")], stop_reason="end_turn"),
        ]
    )

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="garbled ticket text",
        risk_report=_risk_report(),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_forced_scorer(score=0.99, confidence=0.99),
        case_id="c6",
    )

    assert outcome.branch == "fallthrough"
    assert outcome.seed_messages is None
    assert outcome.grounding_tool_calls == []
    # No check_return_policy call was ever made -- extraction failed first.


def test_run_gate_jev_client_error_falls_through_with_grounding_preserved(mat_module):
    class _BoomClient:
        def score(self, query):
            raise RuntimeError("simulated Jev outage")

    client = ScriptedClient([_extraction_response()])

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="My ORD-1001 earbuds arrived cracked, please refund me.",
        risk_report=_risk_report(),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_BoomClient(),
        case_id="c7",
    )

    assert outcome.branch == "fallthrough"
    # The check_return_policy call the gate already made is not wasted.
    assert outcome.grounding_tool_calls[0].name == "check_return_policy"
    assert outcome.seed_seen_calls


def test_run_gate_safety_net_corrects_a_fast_path_that_lies_about_the_outcome(mat_module):
    # The forced write-up call falsely claims REJECTED even though the real,
    # deterministic process_refund call (dispatched by the gate itself, not
    # the model) actually returns APPROVED -- enforce_decision must still
    # catch and correct this, exactly as it does for DecisionAgent's own
    # freeform path. Proves the fast path is not trusted blindly.
    client = ScriptedClient(
        [
            _extraction_response(),
            _submit_response(refund_status="REJECTED", approved_amount=None, refund_id=None),
        ]
    )

    outcome = run_gate(
        client=client,
        model="x",
        ticket_text="My ORD-1001 earbuds arrived cracked, please refund me.",
        risk_report=_risk_report(),
        tool_registry=mat_module.TOOL_REGISTRY,
        jev_client=_forced_scorer(score=0.99, confidence=0.99),
        case_id="c8",
    )

    assert outcome.branch == "fast_approve"
    assert outcome.decision_result.decision.refund_status == "APPROVED"
    assert outcome.decision_result.decision.approved_amount == 35.0
    assert outcome.decision_result.corrections
