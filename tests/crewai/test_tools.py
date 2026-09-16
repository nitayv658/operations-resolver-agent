"""Tests for resolver_agent_crewai.tools.build_tools().

No LLM/crew involved -- these call each CrewAI tool directly, the same way
CrewAI itself would once an agent decides to invoke one, and check that the
real starter-kit dispatch, the ToolCallRecord tracing, and the
cross-customer authorization wrapping all work exactly as they do for Part 1
(tests/test_authorization.py tests authorize_tool_registry itself; this
checks it's wired into the CrewAI tool layer correctly).
"""

from __future__ import annotations

from resolver_agent.tool_loop import ToolCallRecord
from resolver_agent_crewai.tools import build_tools


def test_build_tools_returns_the_four_globalcart_tools_by_name():
    call_log: list = []
    tools = build_tools(call_log, None, {"case_id": "t1"})

    names = {tool.name for tool in tools}
    assert names == {"get_order_details", "get_user_profile", "check_return_policy", "process_refund"}


def test_build_tools_dispatches_to_the_real_starter_kit_function_and_records_the_call():
    call_log: list = []
    tools = build_tools(call_log, None, {"case_id": "t1"})
    order_tool = next(t for t in tools if t.name == "get_order_details")

    result = order_tool.run(order_id="ORD-1001")

    assert result["order_id"] == "ORD-1001"
    assert result["user_id"] == "USR-101"
    assert len(call_log) == 1
    record = call_log[0]
    assert isinstance(record, ToolCallRecord)
    assert record.name == "get_order_details"
    assert record.input == {"order_id": "ORD-1001"}
    assert record.result == result


def test_build_tools_when_requester_id_mismatches_owner_should_deny():
    call_log: list = []
    tools = build_tools(call_log, "USR-999", {"case_id": "t1"})  # ORD-1001 belongs to USR-101
    order_tool = next(t for t in tools if t.name == "get_order_details")

    result = order_tool.run(order_id="ORD-1001")

    assert result == {
        "error": "NOT_AUTHORIZED",
        "message": "This record does not belong to the requesting customer.",
    }
    assert call_log[0].result == result


def test_build_tools_when_requester_id_matches_owner_should_proceed_normally():
    call_log: list = []
    tools = build_tools(call_log, "USR-101", {"case_id": "t1"})
    order_tool = next(t for t in tools if t.name == "get_order_details")

    result = order_tool.run(order_id="ORD-1001")

    assert "error" not in result
    assert result["order_id"] == "ORD-1001"


def test_build_tools_when_omitted_requester_id_should_stay_unrestricted():
    call_log: list = []
    tools = build_tools(call_log, None, {"case_id": "t1"})
    order_tool = next(t for t in tools if t.name == "get_order_details")

    # ORD-1001 belongs to USR-101, not the (nonexistent) caller -- with no
    # requester_user_id bound at all, this must still succeed unrestricted.
    result = order_tool.run(order_id="ORD-1001")

    assert "error" not in result


def test_build_tools_each_call_gets_its_own_call_log_and_registry():
    # CrewResolverAgent.resolve() calls build_tools() fresh per ticket --
    # confirm two builds don't share state (mirrors ResolverAgent.resolve()
    # rebuilding an authorization-wrapped registry fresh every call too).
    log_a: list = []
    log_b: list = []
    tools_a = build_tools(log_a, None, {"case_id": "a"})
    tools_b = build_tools(log_b, None, {"case_id": "b"})

    next(t for t in tools_a if t.name == "get_order_details").run(order_id="ORD-1001")

    assert len(log_a) == 1
    assert len(log_b) == 0


def test_process_refund_args_schema_rejects_a_reason_outside_the_enum():
    call_log: list = []
    tools = build_tools(call_log, None, {"case_id": "t1"})
    refund_tool = next(t for t in tools if t.name == "process_refund")

    args_schema = refund_tool.args_schema
    assert "reason" in args_schema.model_fields
    # mock_services.TOOL_SCHEMAS constrains `reason` to a fixed enum -- the
    # generated pydantic field must reject a value outside it, same as the
    # Anthropic API's own JSON-schema enum would for Part 1.
    import pytest

    with pytest.raises(Exception):
        args_schema(order_id="ORD-1001", amount=10.0, reason="not_a_real_reason")
