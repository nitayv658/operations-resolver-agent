"""Tests for resolver_agent_crewai.crew.tools -- the authority-separation
guarantee (mirrors tests/crew/test_tool_ownership.py's static assertions,
but for the CrewAI tool objects each build_*_tools() actually returns) and
the dispatch-boundary guardrail on send_slack_alert.
"""

from __future__ import annotations

from resolver_agent.tool_loop import ToolCallRecord
from resolver_agent_crewai.crew.tools import (
    build_comms_tools,
    build_decision_tools,
    build_researcher_tools,
)


def _names(tools):
    return {t.name for t in tools}


def test_researcher_tools_cannot_reach_money_or_messaging():
    names = _names(build_researcher_tools([]))
    assert names == {"get_order_details", "get_user_profile", "audit_fraud_risk"}
    assert "process_refund" not in names
    assert "send_slack_alert" not in names


def test_decision_tools_are_the_only_ones_with_process_refund():
    names = _names(build_decision_tools([]))
    assert names == {"check_return_policy", "process_refund"}


def test_comms_tools_cannot_reach_money():
    names = _names(build_comms_tools([], "case-1"))
    assert names == {"get_escalation_route", "send_slack_alert"}
    assert "process_refund" not in names


def test_build_researcher_tools_dispatches_and_traces_the_real_call():
    call_log: list = []
    tools = build_researcher_tools(call_log)
    order_tool = next(t for t in tools if t.name == "get_order_details")

    result = order_tool.run(order_id="ORD-1001")

    assert result["order_id"] == "ORD-1001"
    assert len(call_log) == 1
    assert isinstance(call_log[0], ToolCallRecord)
    assert call_log[0].name == "get_order_details"
    assert call_log[0].result == result


def test_send_slack_alert_is_denied_without_a_prior_route_call_this_same_case():
    call_log: list = []
    tools = build_comms_tools(call_log, "case-1")
    alert_tool = next(t for t in tools if t.name == "send_slack_alert")

    result = alert_tool.run(channel_id="CH-FRAUD", severity="high", payload={"order_id": "ORD-X"})

    assert result == {
        "error": "ALERT_NOT_AUTHORIZED",
        "message": (
            "send_slack_alert was refused: this case's own get_escalation_route "
            "call has not returned escalation_required=true. Do not call "
            "send_slack_alert on a case that resolved cleanly."
        ),
    }


def test_send_slack_alert_is_allowed_after_a_route_call_returns_escalation_required():
    call_log: list = []
    tools = build_comms_tools(call_log, "case-1")
    route_tool = next(t for t in tools if t.name == "get_escalation_route")
    alert_tool = next(t for t in tools if t.name == "send_slack_alert")

    route_result = route_tool.run(
        risk_band="high", requested_amount=300.0, prior_fraud_flags=1, order_status="delivered", verdict="ELIGIBLE"
    )
    assert route_result["escalation_required"] is True

    alert_result = alert_tool.run(
        channel_id=route_result["channel_id"], severity=route_result["severity"], payload={"order_id": "ORD-X"}
    )
    assert alert_result["delivered"] is True


def test_send_slack_alert_guard_is_scoped_to_its_own_case_not_shared():
    # A fresh build_comms_tools() call for a different case must not inherit
    # a prior case's escalation_required=true state -- mirrors
    # resolver_agent.crew.comms.agent._guarded_registry's own "state is
    # local to one call" guarantee.
    tools_a = build_comms_tools([], "case-a")
    route_a = next(t for t in tools_a if t.name == "get_escalation_route")
    route_a.run(risk_band="high", requested_amount=300.0, prior_fraud_flags=1, order_status="delivered", verdict="ELIGIBLE")

    tools_b = build_comms_tools([], "case-b")
    alert_b = next(t for t in tools_b if t.name == "send_slack_alert")
    result = alert_b.run(channel_id="CH-FRAUD", severity="high", payload={"order_id": "ORD-Y"})

    assert result.get("error") == "ALERT_NOT_AUTHORIZED"
