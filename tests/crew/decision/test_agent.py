"""Tests for DecisionAgent.run() -- scripted model turns, dispatched through
the real starter-kit functions. The two guardrails this stage owns:
cross-checking refund_status against the real process_refund result, and
the ORD-1005-style guardrail where risk_report.blocks_automatic_refund
overrides a policy verdict that would otherwise approve."""

from __future__ import annotations

from resolver_agent.crew.decision.agent import DecisionAgent
from resolver_agent.crew.decision.output_tool import SUBMIT_DECISION_TOOL_NAME
from resolver_agent.crew.schemas import RiskReport

from ...helpers import ScriptedClient, ScriptedResponse, tool_use_block


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


def _submit(**overrides):
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
        "rationale": "Order is eligible and within the VIP cap.",
    }
    payload.update(overrides)
    return ScriptedResponse([tool_use_block(SUBMIT_DECISION_TOOL_NAME, payload)])


def test_run_approves_a_clean_case_matching_the_real_process_refund_result():
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1001", "amount": 35.0})]),
            _submit(),
        ]
    )
    agent = DecisionAgent(client=client, model="x")

    result = agent.run(_risk_report(), case_id="c1")

    assert result.error is None
    assert result.decision.refund_status == "APPROVED"
    assert result.decision.approved_amount == 35.0
    assert result.corrections == []


def test_run_overrides_approval_that_disagrees_with_process_refund():
    # ORD-1002's espresso machine claim ($150) is genuinely above the cap --
    # process_refund will return ESCALATION_REQUIRED regardless of what the
    # model claims.
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1002"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1002", "amount": 150.0})]),
            _submit(
                order_id="ORD-1002",
                user_id="USR-102",
                requested_amount=150.0,
                approved_amount=150.0,
                refund_id="RF-FAKE",
                refund_status="APPROVED",  # false -- process_refund actually escalates
            ),
        ]
    )
    agent = DecisionAgent(client=client, model="x")

    result = agent.run(_risk_report(order_id="ORD-1002", user_id="USR-102"), case_id="c2")

    assert result.decision.refund_status == "ESCALATION_REQUIRED"
    assert result.decision.approved_amount is None
    assert result.corrections


def test_run_overrides_an_under_request_that_dodges_escalation():
    # ORD-1002's real claim is $150, above the $50 Standard cap. Requesting
    # exactly $50 gets a clean APPROVED back from process_refund (it
    # enforces its cap, not intent) -- only the risk report's
    # evidence.order_total_usd (the real amount owed, passed through from
    # the Researcher) can tell this apart from an honest $50 claim.
    risk_report = _risk_report(
        order_id="ORD-1002",
        user_id="USR-102",
        evidence={"order_total_usd": 150.0, "order_status": "delivered", "prior_fraud_flags": 0},
    )
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1002"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1002", "amount": 50.0})]),
            _submit(
                order_id="ORD-1002",
                user_id="USR-102",
                requested_amount=50.0,
                approved_amount=50.0,
                refund_id="RF-1002-5000",
                refund_status="APPROVED",
            ),
        ]
    )
    agent = DecisionAgent(client=client, model="x")

    result = agent.run(risk_report, case_id="c4")

    assert result.decision.refund_status == "ESCALATION_REQUIRED"
    assert result.decision.approved_amount is None
    assert result.decision.refund_id is None
    # requested_amount must also be corrected to the real amount owed --
    # otherwise Comms's get_escalation_route call would see the
    # under-requested $50 (at the cap, nothing to escalate) instead of the
    # real $150, and silently send no alert despite the override above.
    assert result.decision.requested_amount == 150.0
    assert any("under-request" in w for w in result.warnings)


def test_run_blocks_approval_when_risk_report_says_blocks_automatic_refund():
    # Isolates the ORD-1005-style guardrail from the process_refund
    # cross-check above: here process_refund would legitimately approve
    # (amount is well within the VIP cap), but the incoming risk report
    # says this case is high-risk and blocks_automatic_refund=True. The
    # fraud finding must win even though the tool itself said APPROVED.
    high_risk_report = _risk_report(risk_score=90, risk_band="high", blocks_automatic_refund=True)
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1001", "amount": 35.0})]),
            _submit(refund_status="APPROVED"),
        ]
    )
    agent = DecisionAgent(client=client, model="x")

    result = agent.run(high_risk_report, case_id="c3")

    assert result.decision.refund_status == "ESCALATION_REQUIRED"
    assert result.decision.approved_amount is None
    assert result.decision.refund_id is None
    assert any("blocks_automatic_refund" in w for w in result.warnings)
