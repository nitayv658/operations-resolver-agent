"""Tests for ResearcherAgent.run() -- scripted model turns, dispatched
through the real starter-kit/multi_agent_tools functions (no mocking of the
tool box itself, same philosophy as tests/helpers.py's own docstring)."""

from __future__ import annotations

from resolver_agent.crew.researcher.agent import ResearcherAgent
from resolver_agent.crew.researcher.output_tool import SUBMIT_RISK_REPORT_TOOL_NAME

from ...helpers import ScriptedClient, ScriptedResponse, tool_use_block


def _submit_ok(**overrides):
    # Matches the real audit_fraud_risk("ORD-1001", "USR-101") result
    # field-for-field -- a model that transcribed it correctly, so
    # enforce_risk_report should find nothing to correct here.
    payload = {
        "status": "OK",
        "order_id": "ORD-1001",
        "user_id": "USR-101",
        "risk_score": 0,
        "risk_band": "low",
        "action_hint": "proceed with the normal refund flow",
        "triggered_rules": [],
        "evidence": {
            "reference_date": "2026-08-05",
            "order_total_usd": 35.0,
            "order_status": "delivered",
            "claims_in_last_60_days": 0,
            "refunded_usd_in_last_60_days": 0,
            "account_age_days": 1243,
            "prior_fraud_flags": 0,
            "initial_fraud_score": 5,
            "address_changed_at": None,
            "address_change_days_before_delivery": None,
            "missing_item_skus": [],
        },
        "blocks_automatic_refund": False,
        "requires_security_channel": False,
        "rulebook_version": "1.0.0",
    }
    payload.update(overrides)
    return ScriptedResponse([tool_use_block(SUBMIT_RISK_REPORT_TOOL_NAME, payload)])


def test_run_produces_a_risk_report_matching_the_real_audit_result():
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("get_user_profile", {"user_id": "USR-101"})]),
            ScriptedResponse([tool_use_block("audit_fraud_risk", {"order_id": "ORD-1001", "user_id": "USR-101"})]),
            _submit_ok(),
        ]
    )
    agent = ResearcherAgent(client=client, model="x")

    result = agent.run("My order ORD-1001 arrived damaged, please refund me.", case_id="c1")

    assert result.error is None
    assert result.report is not None
    assert result.report.risk_band == "low"
    assert result.report.risk_score == 0
    assert result.corrections == []


def test_run_overrides_a_self_reported_band_that_disagrees_with_the_real_audit():
    # The model calls audit_fraud_risk for the real high-risk order/user, but
    # then mis-transcribes the band as "low" in its submit_risk_report call.
    # enforce_risk_report must catch and correct this -- the fraud engine's
    # own result is the only truth, never the model's paraphrase of it.
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1005"})]),
            ScriptedResponse([tool_use_block("get_user_profile", {"user_id": "USR-105"})]),
            ScriptedResponse([tool_use_block("audit_fraud_risk", {"order_id": "ORD-1005", "user_id": "USR-105"})]),
            _submit_ok(
                order_id="ORD-1005",
                user_id="USR-105",
                risk_score=0,
                risk_band="low",
                blocks_automatic_refund=False,
                requires_security_channel=False,
            ),
        ]
    )
    agent = ResearcherAgent(client=client, model="x")

    result = agent.run("Order ORD-1005, refund me the full amount.", case_id="c2")

    assert result.report is not None
    # Overridden to the real audit_fraud_risk result, not the model's claim.
    assert result.report.risk_band == "high"
    assert result.report.risk_score == 90
    assert result.report.blocks_automatic_refund is True
    assert result.corrections  # the drift was recorded, not silently fixed


def test_run_reports_a_lookup_failure_instead_of_inventing_a_risk_score():
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
    agent = ResearcherAgent(client=client, model="x")

    result = agent.run("My order ORD-9999 never arrived.", case_id="c3")

    assert result.report is None
    assert result.error == {"error": "ORDER_NOT_FOUND", "message": "No order found with id 'ORD-9999'."}
    assert result.order_id == "ORD-9999"


def test_run_treats_no_submit_call_as_a_failure_not_a_hang():
    client = ScriptedClient([ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})])] * 6)
    agent = ResearcherAgent(client=client, model="x", max_iterations=2)

    result = agent.run("Order ORD-1001.", case_id="c4")

    assert result.report is None
    assert result.error is not None
    assert result.stopped_reason == "max_iterations"
