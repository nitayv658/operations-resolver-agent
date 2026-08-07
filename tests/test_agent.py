"""Tests for ResolverAgent.resolve()'s handling of API-level failures.

ResolverAgent is constructed with a fake client (see tests/helpers.py) so
these tests need no ANTHROPIC_API_KEY and make no network call. Only the
model's turns are faked; if a scenario reaches a real tool call, it still
dispatches through the real starter-kit functions.
"""

from __future__ import annotations

import anthropic
import pytest

from resolver_agent import ResolverAgent

from .helpers import (
    ScriptedClient,
    ScriptedResponse,
    fake_api_connection_error,
    fake_bad_request_error,
    tool_use_block,
)


def test_resolve_when_model_api_error_should_return_safe_escalation_with_partial_trace():
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            fake_bad_request_error("Your credit balance is too low to access the Anthropic API."),
        ]
    )
    agent = ResolverAgent(client=client)

    result = agent.resolve("My order ORD-1001 arrived damaged, please refund me.")

    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert result["_stopped_reason"] == "api_error"
    # the tool call made before the API failure must not be lost
    assert result["_tool_calls"][0]["name"] == "get_order_details"
    # the failure itself should be visible in the audit trail, not swallowed
    assert any("BadRequestError" in line or "credit balance" in line for line in result["reasoning_chain"])
    # never claim a refund happened when the case wasn't actually resolved
    assert "refund" not in result["customer_response"].lower() or "unable" in result["customer_response"].lower() or "sorry" in result["customer_response"].lower()


def test_resolve_when_api_error_happens_on_the_first_call_should_still_return_safe_escalation():
    client = ScriptedClient([fake_api_connection_error()])
    agent = ResolverAgent(client=client)

    result = agent.resolve("Hi, I have a problem with my order.")

    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert result["_stopped_reason"] == "api_error"
    assert result["_tool_calls"] == []


def test_resolve_when_a_non_api_error_occurs_should_still_propagate_not_be_swallowed():
    # A plain bug (not an anthropic.APIError) must NOT be silently turned
    # into a customer-facing escalation -- only genuine, expected API/infra
    # failures get the soft landing. Anything else should keep failing loudly.
    class BrokenClient:
        class _Messages:
            def create(self, **kwargs):
                raise RuntimeError("some unrelated programming bug")

        @property
        def messages(self):
            return self._Messages()

    agent = ResolverAgent(client=BrokenClient())

    with pytest.raises(RuntimeError, match="some unrelated programming bug"):
        agent.resolve("Hi, I have a problem with my order.")
