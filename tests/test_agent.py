"""Tests for ResolverAgent.resolve()'s handling of API-level failures.

ResolverAgent is constructed with a fake client (see tests/helpers.py) so
these tests need no ANTHROPIC_API_KEY and make no network call. Only the
model's turns are faked; if a scenario reaches a real tool call, it still
dispatches through the real starter-kit functions.
"""

from __future__ import annotations

import logging

import anthropic
import pytest

from resolver_agent import ResolverAgent
from resolver_agent.output_tool import SUBMIT_RESOLUTION_TOOL_NAME

from .helpers import (
    ScriptedClient,
    ScriptedResponse,
    fake_api_connection_error,
    fake_bad_request_error,
    text_block,
    tool_use_block,
)


def _submit(decision, refund_amount=None, refund_id=None):
    return ScriptedResponse(
        [
            tool_use_block(
                SUBMIT_RESOLUTION_TOOL_NAME,
                {
                    "reasoning_chain": ["..."],
                    "action_taken": {
                        "tools_called": [],
                        "decision": decision,
                        "refund_amount": refund_amount,
                        "refund_id": refund_id,
                    },
                    "customer_response": "...",
                },
            )
        ]
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


def test_resolve_when_successful_should_log_info_with_case_id_and_decision(caplog):
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            _submit("ESCALATION_REQUIRED"),
        ]
    )
    agent = ResolverAgent(client=client)

    with caplog.at_level(logging.INFO, logger="resolver_agent"):
        result = agent.resolve("Hi, I have a problem with ORD-1001.")

    records = [r for r in caplog.records if r.getMessage() == "agent.case_resolved"]
    assert len(records) == 1
    assert records[0].fields["decision"] == "ESCALATION_REQUIRED"
    assert records[0].fields["case_id"] == result["_case_id"]


def test_resolve_when_validation_warnings_present_should_log_warning(caplog):
    # ORD-1011: $52 order, Standard $50 cap -- process_refund called with
    # exactly 50.0 (the cap) instead of the true 52.0 owed triggers the
    # under-request-to-dodge-escalation validator warning.
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1011"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1011", "amount": 50.0})]),
            _submit("AUTO_REFUND_APPROVED", refund_amount=50.0, refund_id="RF-1011-5000"),
        ]
    )
    agent = ResolverAgent(client=client)

    with caplog.at_level(logging.WARNING, logger="resolver_agent"):
        result = agent.resolve("My order ORD-1011 arrived damaged, it cost $52.")

    assert result["_validation_warnings"]
    records = [r for r in caplog.records if r.getMessage() == "agent.validation_warnings"]
    assert len(records) == 1
    assert records[0].fields["case_id"] == result["_case_id"]
    assert records[0].fields["warning_count"] == len(result["_validation_warnings"])


def test_resolve_when_fallback_resolution_used_should_log_warning(caplog):
    # The model produces plain text instead of any tool_use on its very
    # first turn -- run_tool_loop reports this as an ordinary "stop" (it
    # never entered the tool-calling branch), but no submit_resolution call
    # was ever captured, so ResolverAgent still has to fall back safely.
    client = ScriptedClient([ScriptedResponse([text_block("I don't know what to do.")], stop_reason="end_turn")])
    agent = ResolverAgent(client=client)

    with caplog.at_level(logging.WARNING, logger="resolver_agent"):
        result = agent.resolve("Hi, I have a problem.")

    assert result["_stopped_reason"] == "stop"
    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    records = [r for r in caplog.records if r.getMessage() == "agent.fallback_resolution_used"]
    assert len(records) == 1
    assert records[0].fields["case_id"] == result["_case_id"]


def test_resolve_when_api_error_should_log_error(caplog):
    client = ScriptedClient([fake_bad_request_error("Your credit balance is too low.")])
    agent = ResolverAgent(client=client)

    with caplog.at_level(logging.ERROR, logger="resolver_agent"):
        result = agent.resolve("Hi, I have a problem.")

    records = [r for r in caplog.records if r.getMessage() == "agent.api_error"]
    assert len(records) == 1
    assert records[0].fields["case_id"] == result["_case_id"]


def test_resolve_should_include_case_id_in_every_outcome():
    client = ScriptedClient([_submit("ESCALATION_REQUIRED")])
    agent = ResolverAgent(client=client)
    result = agent.resolve("Hi.")
    assert isinstance(result["_case_id"], str) and result["_case_id"]


@pytest.mark.parametrize("bad_value", [0, -1])
def test_resolver_agent_when_max_iterations_is_not_positive_should_raise_at_construction(bad_value):
    # Fail fast at construction time, not on the first .resolve() call --
    # a misconfigured agent should never look like it was built successfully.
    with pytest.raises(ValueError, match="max_iterations"):
        ResolverAgent(client=ScriptedClient([]), max_iterations=bad_value)
