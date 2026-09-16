"""Tests for CrewResolverAgent that don't require a live model call --
construction-time validation and the static fallback/api-failure builders.

These mirror the equivalent ResolverAgent tests in tests/test_agent.py.
Full end-to-end guardrail parity (schema-invalid fallback, decision
correction via enforce_resolution, cross-customer denial through a full
.resolve() call) needs a scripted double at CrewAI's own LLM-call boundary
and is covered instead by the live scripts/run_scenarios_crewai.py
comparison against scripts/run_scenarios.py (see the plan/README).
"""

from __future__ import annotations

import pytest

from resolver_agent.tool_loop import ToolCallRecord
from resolver_agent_crewai.agent import CrewResolverAgent

from ..helpers import fake_bad_request_error


@pytest.mark.parametrize("bad_value", [0, -1])
def test_crew_resolver_agent_when_max_iterations_is_not_positive_should_raise_at_construction(bad_value):
    with pytest.raises(ValueError, match="max_iterations"):
        CrewResolverAgent(max_iterations=bad_value)


def test_resolve_when_require_verified_requester_and_no_requester_id_should_raise_immediately():
    agent = CrewResolverAgent(require_verified_requester=True)

    with pytest.raises(ValueError, match="require_verified_requester"):
        agent.resolve("Hi, I have a problem with my order.")


def test_litellm_model_string_gets_the_anthropic_provider_prefix():
    agent = CrewResolverAgent(model="claude-sonnet-5")
    assert agent.model == "anthropic/claude-sonnet-5"


def test_litellm_model_string_is_left_alone_if_already_prefixed():
    agent = CrewResolverAgent(model="openai/gpt-4o")
    assert agent.model == "openai/gpt-4o"


def test_fallback_resolution_when_no_reason_given_reports_no_structured_output():
    call_log = [ToolCallRecord(name="get_order_details", input={"order_id": "ORD-1001"}, result={"order_id": "ORD-1001"})]

    resolution = CrewResolverAgent._fallback_resolution(call_log)

    assert resolution["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert resolution["action_taken"]["tools_called"] == ["get_order_details"]
    assert "did not produce" in resolution["reasoning_chain"][0]


def test_fallback_resolution_when_schema_errors_given_reports_them():
    resolution = CrewResolverAgent._fallback_resolution([], reason=["decision is missing"])

    assert resolution["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert "structurally invalid" in resolution["reasoning_chain"][0]
    assert "decision is missing" in resolution["reasoning_chain"][0]


def test_api_failure_resolution_reports_the_failure_and_escalates():
    # fake_bad_request_error builds a real anthropic.BadRequestError (see
    # tests/helpers.py) -- the exact type CrewAI's native Anthropic provider
    # actually re-raises (confirmed empirically, see agent.py's docstring),
    # reused here rather than hand-rolling a fake exception.
    exc = fake_bad_request_error("boom")
    call_log = [ToolCallRecord(name="get_order_details", input={"order_id": "ORD-1001"}, result={})]

    resolution = CrewResolverAgent._api_failure_resolution(exc, call_log)

    assert resolution["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert resolution["action_taken"]["tools_called"] == ["get_order_details"]
    assert "boom" in resolution["reasoning_chain"][0]
