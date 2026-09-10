"""Full-pipeline tests for OperationsCrew.handle_ticket -- one ScriptedClient
shared across all three agents (a real crew always shares one client), real
tool dispatch throughout."""

from __future__ import annotations

from resolver_agent.crew import orchestrator
from resolver_agent.crew.comms.output_tool import SUBMIT_COMMS_RESULT_TOOL_NAME
from resolver_agent.crew.decision.output_tool import SUBMIT_DECISION_TOOL_NAME
from resolver_agent.crew.orchestrator import OperationsCrew
from resolver_agent.crew.researcher.output_tool import SUBMIT_RISK_REPORT_TOOL_NAME

from ..helpers import ScriptedClient, ScriptedResponse, text_block, tool_use_block


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


def test_decision_incomplete_on_a_high_risk_report_still_dispatches_a_security_alert():
    """Live behavior this reproduces: the Researcher correctly scores a case
    high risk, but the Decision agent stalls (ends its turn without calling
    submit_decision) before ever routing anything to security. tool_loop's
    own forced-retry (see resolver_agent/tool_loop.py's _forced_stop_call)
    gets one more attempt with tool_choice pinned to submit_decision -- this
    scripts that retry stalling too, so the case still reaches genuine
    decision_incomplete. Without the orchestrator's own fallback, that risk
    finding would be dropped on the floor -- the customer still gets a safe
    reply, but Trust & Safety would never hear about a risk_score=90 case.
    This proves the fallback fires without any further LLM call (client.calls
    stays at exactly the Researcher's 4 + Decision's 2 stalled attempts)."""
    client = ScriptedClient(
        [
            # Researcher -- succeeds, flags high risk
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
            # Decision -- stalls: ends its turn in plain text, never calls submit_decision
            ScriptedResponse([text_block("I don't have enough information to proceed.")], stop_reason="end_turn"),
            # tool_loop's forced retry (tool_choice pinned to submit_decision) -- stalls too
            ScriptedResponse([text_block("Still not sure.")], stop_reason="end_turn"),
        ]
    )
    crew = OperationsCrew(client=client, model="x")

    result = crew.handle_ticket(
        "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me the full 480 dollars."
    )

    assert result.decision is None
    assert result.stopped_reason == "decision_incomplete"
    assert result.alert_sent is True
    assert result.escalation is not None
    assert result.escalation["channel_id"] == "CH-FRAUD"
    assert result.alert_record is not None
    assert result.alert_record["delivered"] is True
    assert any("dispatched a direct security alert" in line for line in result.reasoning_chain)
    assert client.calls == 6  # Researcher's 4 calls + Decision's 2 stalled attempts -- no LLM call for the alert itself


def test_decision_incomplete_on_a_low_risk_report_dispatches_no_alert():
    """Same stall, but on a report that never asked for the security channel
    in the first place -- the fallback must not invent an escalation the
    Researcher never found, matching the prior (no-alert) behavior exactly
    for the common case."""
    client = ScriptedClient(
        [
            # Researcher -- succeeds, low risk
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
            # Decision -- stalls the same way
            ScriptedResponse([text_block("I don't have enough information to proceed.")], stop_reason="end_turn"),
            # tool_loop's forced retry (tool_choice pinned to submit_decision) -- stalls too
            ScriptedResponse([text_block("Still not sure.")], stop_reason="end_turn"),
        ]
    )
    crew = OperationsCrew(client=client, model="x")

    result = crew.handle_ticket("My earbuds from order ORD-1001 arrived cracked, please refund me.")

    assert result.decision is None
    assert result.stopped_reason == "decision_incomplete"
    assert result.alert_sent is False
    assert result.escalation is None
    assert result.alert_record is None
    assert client.calls == 6  # Researcher's 4 calls + Decision's 2 stalled attempts


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


def test_per_agent_model_overrides_reach_their_own_agent_only():
    """researcher_model/decision_model/comms_model are independent overrides
    -- each sub-agent gets its own value, not the shared `model`."""
    crew = OperationsCrew(
        client=ScriptedClient([]),
        model="shared-default",
        researcher_model="researcher-model",
        decision_model="decision-model",
        comms_model="comms-model",
    )

    assert crew.researcher.model == "researcher-model"
    assert crew.decision_agent.model == "decision-model"
    assert crew.comms_agent.model == "comms-model"


def test_model_kwarg_alone_applies_to_researcher_and_decision_but_not_comms():
    """`model=` remains the shared fallback for Researcher/Decision, matching
    every existing call site/test in this file -- but Comms is deliberately
    NOT chained to it any more: it has its own independent default
    (DEFAULT_COMMS_MODEL, Haiku), so passing `model=` alone no longer moves
    Comms too."""
    crew = OperationsCrew(client=ScriptedClient([]), model="x")

    assert crew.researcher.model == "x"
    assert crew.decision_agent.model == "x"
    assert crew.comms_agent.model == orchestrator.DEFAULT_COMMS_MODEL
    assert crew.comms_agent.model != "x"


def test_per_agent_env_vars_apply_when_no_kwarg_is_passed(monkeypatch):
    """The middle tier of the resolution order (kwarg -> env var -> default)
    -- scripts/run_crew.py/scripts/run_crew_scenarios.py both construct OperationsCrew()
    with no per-agent kwargs at all, so this env-var path is the only way
    either entry point can actually reach it."""
    monkeypatch.setenv("ANTHROPIC_MODEL_RESEARCHER", "researcher-from-env")
    monkeypatch.setenv("ANTHROPIC_MODEL_DECISION", "decision-from-env")
    monkeypatch.setenv("ANTHROPIC_MODEL_COMMS", "comms-from-env")

    crew = OperationsCrew(client=ScriptedClient([]), model="shared-default")

    assert crew.researcher.model == "researcher-from-env"
    assert crew.decision_agent.model == "decision-from-env"
    assert crew.comms_agent.model == "comms-from-env"


def test_comms_defaults_to_haiku_with_no_kwarg_model_or_env_var(monkeypatch):
    """The actual default path scripts/run_crew.py/scripts/run_crew_scenarios.py take: no
    per-agent kwargs, no env vars set at all -- Comms should land on
    DEFAULT_COMMS_MODEL (Haiku), not DEFAULT_MODEL (Sonnet), while
    Researcher/Decision stay on DEFAULT_MODEL."""
    monkeypatch.delenv("ANTHROPIC_MODEL_RESEARCHER", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL_DECISION", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL_COMMS", raising=False)

    crew = OperationsCrew(client=ScriptedClient([]))

    assert crew.researcher.model == orchestrator.DEFAULT_MODEL
    assert crew.decision_agent.model == orchestrator.DEFAULT_MODEL
    assert crew.comms_agent.model == orchestrator.DEFAULT_COMMS_MODEL
    assert orchestrator.DEFAULT_COMMS_MODEL != orchestrator.DEFAULT_MODEL


def test_explicit_kwarg_wins_over_env_var(monkeypatch):
    """Completes the precedence chain: an explicit kwarg beats its own env
    var, even when both are set for the same agent."""
    monkeypatch.setenv("ANTHROPIC_MODEL_DECISION", "decision-from-env")

    crew = OperationsCrew(
        client=ScriptedClient([]),
        model="shared-default",
        decision_model="decision-from-kwarg",
    )

    assert crew.decision_agent.model == "decision-from-kwarg"
