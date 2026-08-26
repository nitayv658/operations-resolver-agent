"""Full-pipeline tests for OperationsCrew.handle_ticket -- one ScriptedClient
shared across all three agents (a real crew always shares one client), real
tool dispatch throughout."""

from __future__ import annotations

from resolver_agent.crew.comms.output_tool import SUBMIT_COMMS_RESULT_TOOL_NAME
from resolver_agent.crew.decision.output_tool import SUBMIT_DECISION_TOOL_NAME
from resolver_agent.crew.orchestrator import OperationsCrew
from resolver_agent.crew.researcher.output_tool import SUBMIT_RISK_REPORT_TOOL_NAME

from ..helpers import ScriptedClient, ScriptedResponse, tool_use_block


def test_ord_1005_trap_is_blocked_even_with_a_falsely_approving_decision():
    """The claim is genuinely policy-eligible; only the fraud report should
    stop the payout. Here the Decision agent's own scripted call falsely
    claims APPROVED -- the crew must still end up ESCALATION_REQUIRED with
    a real alert sent, proving the guardrail survives end to end, not only
    in DecisionAgent's own unit test."""
    client = ScriptedClient(
        [
            # Researcher
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1005"})]),
            ScriptedResponse([tool_use_block("get_user_profile", {"user_id": "USR-105"})]),
            ScriptedResponse([tool_use_block("audit_fraud_risk", {"order_id": "ORD-1005", "user_id": "USR-105"})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RISK_REPORT_TOOL_NAME,
                        {
                            "status": "OK",
                            "order_id": "ORD-1005",
                            "user_id": "USR-105",
                            "risk_score": 90,
                            "risk_band": "high",
                            "action_hint": "block the automatic refund and escalate to the security channel",
                            "triggered_rules": [{"rule_id": "FR-01", "name": "repeat_refund_claims", "weight": 25, "why": "..."}],
                            "evidence": {"order_total_usd": 480.0, "order_status": "delivered", "prior_fraud_flags": 1},
                            "blocks_automatic_refund": True,
                            "requires_security_channel": True,
                            "rulebook_version": "1.0.0",
                        },
                    )
                ]
            ),
            # Decision -- calls the real tools, then falsely claims APPROVED
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1005"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1005", "amount": 480.0})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_DECISION_TOOL_NAME,
                        {
                            "order_id": "ORD-1005",
                            "user_id": "USR-105",
                            "verdict": "ELIGIBLE",
                            "eligible": True,
                            "refund_status": "APPROVED",  # false -- see assertions
                            "requested_amount": 480.0,
                            "approved_amount": 480.0,
                            "refund_id": "RF-FAKE",
                            "applicable_policies": ["POL-RET-01"],
                            "rationale": "Claim is eligible under policy.",
                        },
                    )
                ]
            ),
            # Comms
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
                            "payload": {"order_id": "ORD-1005", "user_id": "USR-105", "risk_score": 90, "risk_band": "high"},
                        },
                    )
                ]
            ),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_COMMS_RESULT_TOOL_NAME,
                        {"customer_response": "Your request is being reviewed and we'll follow up shortly."},
                    )
                ]
            ),
        ]
    )
    crew = OperationsCrew(client=client, model="x")

    result = crew.handle_ticket(
        "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me the full 480 dollars."
    )

    assert result.decision is not None
    assert result.decision.refund_status == "ESCALATION_REQUIRED"
    assert result.decision.approved_amount is None
    assert result.alert_sent is True
    assert result.alert_record["channel_id"] == "CH-FRAUD"
    assert any("blocks_automatic_refund" in line for line in result.reasoning_chain)


def test_researcher_lookup_failure_escalates_without_a_second_researcher_call():
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-9999"})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RISK_REPORT_TOOL_NAME,
                        {
                            "status": "LOOKUP_FAILED",
                            "order_id": "ORD-9999",
                            "error": {"error": "ORDER_NOT_FOUND", "message": "No order found with id 'ORD-9999'."},
                        },
                    )
                ]
            ),
        ]
    )
    crew = OperationsCrew(client=client, model="x")

    result = crew.handle_ticket("My order ORD-9999 never arrived and I want the $300 back.")

    assert result.decision is None
    assert result.stopped_reason == "researcher_incomplete"
    # Exactly the Researcher's own two calls -- no re-dispatch, and no
    # Decision/Comms calls made on an incomplete report.
    assert client.calls == 2


def test_clean_case_never_dispatches_a_real_alert_even_if_the_model_tries():
    client = ScriptedClient(
        [
            # Researcher
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("get_user_profile", {"user_id": "USR-101"})]),
            ScriptedResponse([tool_use_block("audit_fraud_risk", {"order_id": "ORD-1001", "user_id": "USR-101"})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RISK_REPORT_TOOL_NAME,
                        {
                            "status": "OK",
                            "order_id": "ORD-1001",
                            "user_id": "USR-101",
                            "risk_score": 0,
                            "risk_band": "low",
                            "action_hint": "proceed with the normal refund flow",
                            "triggered_rules": [],
                            "evidence": {"order_total_usd": 35.0, "order_status": "delivered", "prior_fraud_flags": 0},
                            "blocks_automatic_refund": False,
                            "requires_security_channel": False,
                            "rulebook_version": "1.0.0",
                        },
                    )
                ]
            ),
            # Decision
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1001", "amount": 35.0})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_DECISION_TOOL_NAME,
                        {
                            "order_id": "ORD-1001",
                            "user_id": "USR-101",
                            "verdict": "ELIGIBLE",
                            "eligible": True,
                            "refund_status": "APPROVED",
                            "requested_amount": 35.0,
                            "approved_amount": 35.0,
                            "refund_id": "RF-1001-3500",
                            "applicable_policies": ["POL-RET-02"],
                            "rationale": "Eligible and within the VIP cap.",
                        },
                    )
                ]
            ),
            # Comms -- an over-eager attempt to alert on a clean case, without
            # ever checking the route first.
            ScriptedResponse(
                [
                    tool_use_block(
                        "send_slack_alert",
                        {"channel_id": "CH-FRAUD", "severity": "critical", "payload": {"order_id": "ORD-1001"}},
                    )
                ]
            ),
            ScriptedResponse(
                [tool_use_block(SUBMIT_COMMS_RESULT_TOOL_NAME, {"customer_response": "Your refund has been approved."})]
            ),
        ]
    )
    crew = OperationsCrew(client=client, model="x")

    result = crew.handle_ticket("My earbuds from order ORD-1001 arrived cracked, please refund me.")

    assert result.decision.refund_status == "APPROVED"
    assert result.alert_sent is False
    assert result.alert_record is None
