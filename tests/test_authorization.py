"""Tests for resolver_agent.authorization.authorize_tool_registry.

Dispatched through the real starter-kit functions (gc.TOOL_REGISTRY), not a
mock of them -- same "never mock the tool box itself" philosophy
tests/helpers.py's own docstring states. ORD-1001 belongs to USR-101 (Maya)
in the real starter-kit fixtures.
"""

from __future__ import annotations

from resolver_agent.agent import gc
from resolver_agent.authorization import authorize_tool_registry


def test_authorize_tool_registry_when_owner_matches_requester_should_pass_through_real_data():
    registry = authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-101", log_context={})
    result = registry["get_order_details"](order_id="ORD-1001")

    assert result["order_id"] == "ORD-1001"
    assert "error" not in result


def test_authorize_tool_registry_when_owner_does_not_match_requester_should_deny():
    registry = authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-999", log_context={})
    result = registry["get_order_details"](order_id="ORD-1001")  # actually belongs to USR-101

    assert result == {
        "error": "NOT_AUTHORIZED",
        "message": "This record does not belong to the requesting customer.",
    }


def test_authorize_tool_registry_should_deny_across_all_four_tools_not_just_lookups():
    # Every GlobalCart tool's successful result carries a user_id field --
    # confirmed by reading mock_services.py -- so the same protection must
    # cover check_return_policy/process_refund, not just the two obvious
    # lookup tools.
    registry = authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-999", log_context={})

    assert registry["get_user_profile"](user_id="USR-101")["error"] == "NOT_AUTHORIZED"
    assert registry["check_return_policy"](order_id="ORD-1001")["error"] == "NOT_AUTHORIZED"
    assert registry["process_refund"](order_id="ORD-1001", amount=35.0)["error"] == "NOT_AUTHORIZED"


def test_authorize_tool_registry_should_not_mask_genuine_tool_errors():
    # A nonexistent order must still surface as ORDER_NOT_FOUND, not get
    # relabeled as an authorization failure.
    registry = authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-101", log_context={})
    result = registry["get_order_details"](order_id="ORD-9999")

    assert result["error"] == "ORDER_NOT_FOUND"
