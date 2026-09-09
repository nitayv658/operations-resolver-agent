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
    assert any("cap leaking into requested_amount" in w for w in result.warnings)


def test_run_corrects_a_self_reported_requested_amount_that_leaked_the_cap():
    # Observed live on ORD-1012: blocks_automatic_refund=true correctly
    # meant process_refund was never called at all -- but the model
    # self-reported requested_amount as the $50 auto-refund cap it had just
    # seen from check_return_policy, instead of the risk report's real $890
    # order total. Unlike test_run_overrides_an_under_request_that_dodges_
    # escalation above, there's no process_refund call to cross-check
    # requested_amount against here -- one of three live-observed shapes
    # this same leak takes; see the other two below.
    risk_report = _risk_report(
        order_id="ORD-1012",
        user_id="USR-109",
        risk_score=60,
        risk_band="high",
        blocks_automatic_refund=True,
        requires_security_channel=True,
        evidence={"order_total_usd": 890.0, "order_status": "delivered", "prior_fraud_flags": 0},
    )
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1012"})]),
            _submit(
                order_id="ORD-1012",
                user_id="USR-109",
                requested_amount=50.0,  # the cap -- wrong, the real claim is $890
                approved_amount=None,
                refund_id=None,
                refund_status="ESCALATION_REQUIRED",  # already correct on its own
            ),
        ]
    )
    agent = DecisionAgent(client=client, model="x")

    result = agent.run(risk_report, case_id="c5")

    assert result.decision.refund_status == "ESCALATION_REQUIRED"
    assert result.decision.requested_amount == 890.0
    assert any("cap leaking into requested_amount" in w for w in result.warnings)
    assert any("requested_amount corrected to 890.0" in c for c in result.corrections)


def test_run_corrects_requested_amount_even_when_process_refund_correctly_escalates():
    # The third live-observed shape, and the one the first version of this
    # fix missed entirely -- caught live on ORD-1005 right after that first
    # version shipped. check_return_policy's own requires_escalation is
    # true here (fraud score 61 >= the POL-ESC-01 threshold, 1 prior fraud
    # flag, 3 claims in 60 days), so process_refund IS called with the
    # capped $50 and correctly returns ESCALATION_REQUIRED on its own --
    # not a wrong APPROVED. refund_status ends up right by coincidence, so
    # neither the tool_status-mismatch check nor the old APPROVED-only
    # under-request check ever looked at requested_amount here -- it rode
    # through to Comms's alert payload still saying $50 on a $480 claim.
    # Real fixture data (ORD-1005/USR-105), not a fake -- process_refund is
    # dispatched for real, and only genuinely returns ESCALATION_REQUIRED
    # here because check_return_policy's requires_escalation is real too.
    risk_report = _risk_report(
        order_id="ORD-1005",
        user_id="USR-105",
        risk_score=90,
        risk_band="high",
        blocks_automatic_refund=True,
        requires_security_channel=True,
        evidence={"order_total_usd": 480.0, "order_status": "delivered", "prior_fraud_flags": 1},
    )
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1005"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1005", "amount": 50.0})]),
            _submit(
                order_id="ORD-1005",
                user_id="USR-105",
                requested_amount=50.0,  # the cap -- wrong, the real claim is $480
                approved_amount=None,
                refund_id=None,
                refund_status="ESCALATION_REQUIRED",
            ),
        ]
    )
    agent = DecisionAgent(client=client, model="x")

    result = agent.run(risk_report, case_id="c7")

    assert result.decision.refund_status == "ESCALATION_REQUIRED"
    assert result.decision.requested_amount == 480.0
    assert any("cap leaking into requested_amount" in w for w in result.warnings)
    assert any("requested_amount corrected to 480.0" in c for c in result.corrections)


def test_run_leaves_a_legitimate_partial_requested_amount_alone():
    # A genuinely partial claim (below the order total, but NOT equal to
    # the cap) must not be "corrected" up to the full order total -- the
    # new check's whole point is catching the cap leaking in, not
    # second-guessing every requested_amount below order_total.
    risk_report = _risk_report(
        order_id="ORD-1012",
        user_id="USR-109",
        blocks_automatic_refund=True,
        requires_security_channel=True,
        evidence={"order_total_usd": 890.0, "order_status": "delivered", "prior_fraud_flags": 0},
    )
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1012"})]),
            _submit(
                order_id="ORD-1012",
                user_id="USR-109",
                requested_amount=200.0,  # a real partial claim, not the $50 cap
                approved_amount=None,
                refund_id=None,
                refund_status="ESCALATION_REQUIRED",
            ),
        ]
    )
    agent = DecisionAgent(client=client, model="x")

    result = agent.run(risk_report, case_id="c6")

    assert result.decision.requested_amount == 200.0
    assert result.corrections == []


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
