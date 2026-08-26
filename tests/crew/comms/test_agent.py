"""Tests for CommsAgent.run() -- scripted model turns, dispatched through
the real starter-kit functions. The guardrail this stage owns: send_slack_alert
can only dispatch for real after this same case's own get_escalation_route
call returned escalation_required=true -- enforced at the tool-dispatch
boundary, not trusted to the prompt alone."""

from __future__ import annotations

from resolver_agent.crew.comms.agent import CommsAgent
from resolver_agent.crew.comms.output_tool import SUBMIT_COMMS_RESULT_TOOL_NAME
from resolver_agent.crew.schemas import Decision, RiskReport

from ...helpers import ScriptedClient, ScriptedResponse, tool_use_block


def _decision(**overrides) -> Decision:
    risk_report = RiskReport(
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
    base = dict(
        order_id="ORD-1001",
        user_id="USR-101",
        verdict="ELIGIBLE",
        eligible=True,
        refund_status="APPROVED",
        requested_amount=35.0,
        approved_amount=35.0,
        refund_id="RF-1001-3500",
        applicable_policies=["POL-RET-02"],
        rationale="Eligible and within the VIP cap.",
        risk_report=risk_report,
    )
    base.update(overrides)
    return Decision(**base)


def _submit_reply(text="Good news -- your refund has been approved."):
    return ScriptedResponse([tool_use_block(SUBMIT_COMMS_RESULT_TOOL_NAME, {"customer_response": text})])


def test_run_sends_no_alert_on_a_clean_case():
    client = ScriptedClient(
        [
            ScriptedResponse(
                [
                    tool_use_block(
                        "get_escalation_route",
                        {"risk_band": "low", "requested_amount": 35.0, "prior_fraud_flags": 0, "order_status": "delivered", "verdict": "ELIGIBLE"},
                    )
                ]
            ),
            _submit_reply(),
        ]
    )
    agent = CommsAgent(client=client, model="x")

    result = agent.run(_decision(), case_id="c1")

    assert result.escalation is not None
    assert result.escalation["escalation_required"] is False
    assert result.alert_sent is False
    assert result.alert_record is None
    assert "approved" in result.customer_response.lower()


def test_send_slack_alert_is_refused_without_a_prior_true_escalation_route():
    # The model tries to alert without ever checking routing first (or
    # after a False result) -- the dispatch-boundary guard must refuse it,
    # not merely rely on the prompt telling it not to.
    client = ScriptedClient(
        [
            ScriptedResponse(
                [
                    tool_use_block(
                        "send_slack_alert",
                        {"channel_id": "CH-FRAUD", "severity": "critical", "payload": {"order_id": "ORD-1001"}},
                    )
                ]
            ),
            _submit_reply(),
        ]
    )
    agent = CommsAgent(client=client, model="x")

    result = agent.run(_decision(), case_id="c2")

    assert result.alert_sent is False
    assert result.alert_record is None
    denied_calls = [c for c in result.tool_calls if c.name == "send_slack_alert"]
    assert denied_calls and denied_calls[0].result.get("error") == "ALERT_NOT_AUTHORIZED"


def test_run_sends_a_real_alert_when_escalation_is_required():
    high_risk_decision = _decision(
        refund_status="ESCALATION_REQUIRED",
        approved_amount=None,
        refund_id=None,
        risk_report=RiskReport(
            order_id="ORD-1005",
            user_id="USR-105",
            risk_score=90,
            risk_band="high",
            action_hint="block the automatic refund and escalate to the security channel",
            triggered_rules=[{"rule_id": "FR-01", "name": "repeat_refund_claims", "weight": 25, "why": "..."}],
            evidence={"order_total_usd": 480.0, "order_status": "delivered", "prior_fraud_flags": 1},
            blocks_automatic_refund=True,
            requires_security_channel=True,
            rulebook_version="1.0.0",
        ),
        order_id="ORD-1005",
        user_id="USR-105",
        requested_amount=480.0,
    )
    client = ScriptedClient(
        [
            ScriptedResponse(
                [
                    tool_use_block(
                        "get_escalation_route",
                        {"risk_band": "high", "requested_amount": 480.0, "prior_fraud_flags": 1, "order_status": "delivered", "verdict": "ELIGIBLE"},
                    )
                ]
            ),
            ScriptedResponse(
                [
                    tool_use_block(
                        "send_slack_alert",
                        {
                            "channel_id": "CH-FRAUD",
                            "severity": "critical",
                            "payload": {
                                "order_id": "ORD-1005",
                                "user_id": "USR-105",
                                "risk_score": 90,
                                "risk_band": "high",
                                "requested_amount": 480.0,
                            },
                        },
                    )
                ]
            ),
            _submit_reply("Your request is being reviewed and we'll follow up shortly."),
        ]
    )
    agent = CommsAgent(client=client, model="x")

    result = agent.run(high_risk_decision, case_id="c3")

    assert result.escalation["channel_id"] == "CH-FRAUD"
    assert result.alert_sent is True
    assert result.alert_record["delivered"] is True
    assert result.alert_record["channel_id"] == "CH-FRAUD"
    # never leaks the fraud flag to the customer
    lowered = result.customer_response.lower()
    assert "fraud" not in lowered and "risk" not in lowered
