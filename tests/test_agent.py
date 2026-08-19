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


def test_resolve_when_decision_contradicts_tool_result_should_correct_it_and_log_warning(caplog):
    # ORD-1011: $52 order, Standard $50 cap -- process_refund called with
    # exactly 50.0 (the cap) instead of the true 52.0 owed: the exact live
    # under-request-to-dodge-escalation bug. resolve() must not just flag
    # this -- it must deterministically correct the returned decision so
    # the customer never sees the wrong outcome (output_tool.enforce_resolution).
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

    # the model's original (wrong) claim is still visible in _validation_warnings...
    assert result["_validation_warnings"]
    assert any("under-requested the refund" in w for w in result["_validation_warnings"])
    # ...but the RETURNED decision and customer-facing text reflect the correction, not the claim
    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert result["action_taken"]["refund_amount"] is None
    assert "escalated" in result["customer_response"].lower()
    assert result["_corrections"]

    records = [r for r in caplog.records if r.getMessage() == "agent.resolution_corrected"]
    assert len(records) == 1
    assert records[0].fields["case_id"] == result["_case_id"]
    assert records[0].fields["correction_count"] == len(result["_corrections"])
    assert records[0].fields["decision"] == "ESCALATION_REQUIRED"


def test_resolve_when_submit_resolution_call_is_schema_invalid_should_fall_back_safely(caplog):
    # A submit_resolution call with an out-of-enum decision -- the exact
    # gap proven during the architecture review: previously this would have
    # passed through validate_resolution with zero warnings. Now it must be
    # treated exactly like "no call at all" and fall back safely.
    client = ScriptedClient(
        [
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RESOLUTION_TOOL_NAME,
                        {
                            "reasoning_chain": ["..."],
                            "action_taken": {"tools_called": [], "decision": "MAYBE_REFUND_LATER"},
                            "customer_response": "...",
                        },
                    )
                ]
            )
        ]
    )
    agent = ResolverAgent(client=client)

    with caplog.at_level(logging.WARNING, logger="resolver_agent"):
        result = agent.resolve("Hi, I have a problem with my order.")

    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert "structurally invalid" in result["reasoning_chain"][0]
    assert "MAYBE_REFUND_LATER" in result["reasoning_chain"][0]

    records = [r for r in caplog.records if r.getMessage() == "agent.fallback_resolution_used"]
    assert len(records) == 1
    assert records[0].fields["schema_errors"] >= 1


def test_resolve_when_submit_resolution_call_is_missing_customer_response_should_fall_back_safely():
    # The other empirical review repro: a resolution missing required
    # fields entirely must not flow through as an apparently clean result.
    client = ScriptedClient(
        [
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RESOLUTION_TOOL_NAME,
                        {
                            "reasoning_chain": ["..."],
                            "action_taken": {"tools_called": [], "decision": "ESCALATION_REQUIRED"},
                            # customer_response deliberately omitted
                        },
                    )
                ]
            )
        ]
    )
    agent = ResolverAgent(client=client)
    result = agent.resolve("Hi, I have a problem with my order.")

    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert isinstance(result["customer_response"], str) and result["customer_response"]


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


def test_resolve_when_decision_is_terminal_should_not_trigger_workflow(tmp_path):
    # REJECTED here is backed by a real check_return_policy call (ORD-1003 is
    # genuinely outside the return window) -- a properly corroborated
    # terminal decision, not the unbacked-REJECTED case the no-corroboration
    # guardrail in output_tool.py now separately catches and escalates.
    client = ScriptedClient(
        [
            ScriptedResponse(
                [tool_use_block("check_return_policy", {"order_id": "ORD-1003", "reason": "changed_mind"})]
            ),
            _submit("REJECTED"),
        ]
    )
    queue_path = tmp_path / "queue.jsonl"
    agent = ResolverAgent(client=client, escalation_queue_path=queue_path)

    result = agent.resolve("Hi, I have a problem.")

    assert result["action_taken"]["decision"] == "REJECTED"
    assert result["_workflow_triggered"] is False
    assert not queue_path.exists()


def test_resolve_when_rejected_with_no_corroborating_tool_call_should_escalate_and_trigger_workflow(tmp_path):
    # The exact scenario the new output_tool.py guardrail exists for: a
    # REJECTED claim with zero backing tool evidence must not reach the
    # customer as-is -- it gets corrected to ESCALATION_REQUIRED, same as
    # any other decision/response-gap violation, and that now-escalated
    # case correctly triggers the ops workflow.
    client = ScriptedClient([_submit("REJECTED")])
    queue_path = tmp_path / "queue.jsonl"
    agent = ResolverAgent(client=client, escalation_queue_path=queue_path)

    result = agent.resolve("Hi, I have a problem.")

    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert any("no tool evidence" in w for w in result["_validation_warnings"])
    assert result["_corrections"]
    assert result["_workflow_triggered"] is True
    assert queue_path.exists()


@pytest.mark.parametrize("decision", ["ESCALATION_REQUIRED", "CANNOT_RESOLVE"])
def test_resolve_when_decision_needs_a_human_should_trigger_workflow_and_log(tmp_path, caplog, decision):
    client = ScriptedClient([_submit(decision)])
    queue_path = tmp_path / "queue.jsonl"
    agent = ResolverAgent(client=client, escalation_queue_path=queue_path)

    with caplog.at_level(logging.INFO, logger="resolver_agent"):
        result = agent.resolve("Hi, I have a problem.")

    assert result["_workflow_triggered"] is True
    assert queue_path.exists()

    records = [r for r in caplog.records if r.getMessage() == "agent.workflow_triggered"]
    assert len(records) == 1
    assert records[0].fields["decision"] == decision
    assert records[0].fields["case_id"] == result["_case_id"]


def test_resolve_when_escalation_writer_is_overridden_should_use_it_instead_of_the_default(tmp_path):
    # A caller wiring in a real ticketing SDK bypasses the generic webhook
    # path entirely via escalation_writer -- confirm ResolverAgent actually
    # threads it through to trigger_workflow rather than ignoring it.
    client = ScriptedClient([_submit("ESCALATION_REQUIRED")])
    written = []
    agent = ResolverAgent(
        client=client,
        escalation_queue_path=tmp_path / "unused.jsonl",
        escalation_writer=lambda record, path: written.append(record),
    )

    result = agent.resolve("Hi, I have a problem.")

    assert result["_workflow_triggered"] is True
    assert len(written) == 1
    assert written[0]["case_id"] == result["_case_id"]
    assert not (tmp_path / "unused.jsonl").exists()


def test_resolve_when_api_error_forces_escalation_should_also_trigger_workflow(tmp_path):
    # The api_error fallback and the normal flow both funnel through
    # _finalize_workflow -- an infra failure that safely escalates still
    # needs a real ops-queue record, not just a customer-facing sentence.
    client = ScriptedClient([fake_bad_request_error("Your credit balance is too low.")])
    queue_path = tmp_path / "queue.jsonl"
    agent = ResolverAgent(client=client, escalation_queue_path=queue_path)

    result = agent.resolve("Hi, I have a problem.")

    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert result["_workflow_triggered"] is True
    assert queue_path.exists()


@pytest.mark.parametrize("bad_value", [0, -1])
def test_resolver_agent_when_max_iterations_is_not_positive_should_raise_at_construction(bad_value):
    # Fail fast at construction time, not on the first .resolve() call --
    # a misconfigured agent should never look like it was built successfully.
    with pytest.raises(ValueError, match="max_iterations"):
        ResolverAgent(client=ScriptedClient([]), max_iterations=bad_value)


# --------------------------------------------------------------------------- #
# requester_user_id / cross-customer authorization. ORD-1001 belongs to
# USR-101 (Maya) in the real starter-kit fixtures.
# --------------------------------------------------------------------------- #


def test_authorize_tool_registry_when_owner_matches_requester_should_pass_through_real_data():
    from resolver_agent.agent import _authorize_tool_registry, gc

    registry = _authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-101", log_context={})
    result = registry["get_order_details"](order_id="ORD-1001")

    assert result["order_id"] == "ORD-1001"
    assert "error" not in result


def test_authorize_tool_registry_when_owner_does_not_match_requester_should_deny():
    from resolver_agent.agent import _authorize_tool_registry, gc

    registry = _authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-999", log_context={})
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
    from resolver_agent.agent import _authorize_tool_registry, gc

    registry = _authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-999", log_context={})

    assert registry["get_user_profile"](user_id="USR-101")["error"] == "NOT_AUTHORIZED"
    assert registry["check_return_policy"](order_id="ORD-1001")["error"] == "NOT_AUTHORIZED"
    assert registry["process_refund"](order_id="ORD-1001", amount=35.0)["error"] == "NOT_AUTHORIZED"


def test_authorize_tool_registry_should_not_mask_genuine_tool_errors():
    # A nonexistent order must still surface as ORDER_NOT_FOUND, not get
    # relabeled as an authorization failure.
    from resolver_agent.agent import _authorize_tool_registry, gc

    registry = _authorize_tool_registry(dict(gc.TOOL_REGISTRY), requester_user_id="USR-101", log_context={})
    result = registry["get_order_details"](order_id="ORD-9999")

    assert result["error"] == "ORDER_NOT_FOUND"


def test_resolve_when_requester_user_id_matches_order_owner_should_proceed_normally():
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            _submit("ESCALATION_REQUIRED"),
        ]
    )
    agent = ResolverAgent(client=client)
    result = agent.resolve("My order ORD-1001 arrived damaged.", requester_user_id="USR-101")

    order_call = next(c for c in result["_tool_calls"] if c["name"] == "get_order_details")
    assert "error" not in order_call["result"]
    assert order_call["result"]["order_id"] == "ORD-1001"


def test_resolve_when_requester_user_id_does_not_match_order_owner_should_deny_and_log(caplog):
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),  # belongs to USR-101
            _submit("CANNOT_RESOLVE"),
        ]
    )
    agent = ResolverAgent(client=client)

    with caplog.at_level(logging.WARNING, logger="resolver_agent"):
        result = agent.resolve("My order ORD-1001 arrived damaged.", requester_user_id="USR-999")

    order_call = next(c for c in result["_tool_calls"] if c["name"] == "get_order_details")
    assert order_call["result"] == {
        "error": "NOT_AUTHORIZED",
        "message": "This record does not belong to the requesting customer.",
    }

    records = [r for r in caplog.records if r.getMessage() == "agent.unauthorized_tool_result_denied"]
    assert len(records) == 1
    assert records[0].fields["tool"] == "get_order_details"
    assert records[0].fields["requester_user_id"] == "USR-999"
    assert records[0].fields["case_id"] == result["_case_id"]


def test_resolve_when_requester_user_id_omitted_should_stay_unrestricted():
    # Regression: the default (no requester binding) must behave exactly as
    # it did before this change -- unrestricted access, every existing
    # caller (run_scenarios.py, run_ticket.py, every other test in this
    # suite) keeps working unchanged.
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            _submit("ESCALATION_REQUIRED"),
        ]
    )
    agent = ResolverAgent(client=client)
    result = agent.resolve("My order ORD-1001 arrived damaged.")

    order_call = next(c for c in result["_tool_calls"] if c["name"] == "get_order_details")
    assert "error" not in order_call["result"]


# --------------------------------------------------------------------------- #
# require_verified_requester -- a deployment-level fail-closed switch, not a
# real authentication mechanism (this package still never verifies
# requester_user_id is true -- see the docstring on resolve()). It only
# guards against a customer-facing deployment silently falling back to
# unrestricted mode because some call site forgot to pass an identity.
# --------------------------------------------------------------------------- #


def test_resolve_when_require_verified_requester_and_no_requester_id_should_raise_immediately():
    agent = ResolverAgent(client=ScriptedClient([]), require_verified_requester=True)

    with pytest.raises(ValueError, match="require_verified_requester"):
        agent.resolve("Hi, I have a problem.")

    # fails before ever calling the model
    assert agent.client.calls == 0


def test_resolve_when_require_verified_requester_and_requester_id_given_should_proceed_normally():
    client = ScriptedClient(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            _submit("ESCALATION_REQUIRED"),
        ]
    )
    agent = ResolverAgent(client=client, require_verified_requester=True)

    result = agent.resolve("My order ORD-1001 arrived damaged.", requester_user_id="USR-101")

    order_call = next(c for c in result["_tool_calls"] if c["name"] == "get_order_details")
    assert "error" not in order_call["result"]


def test_resolve_when_require_verified_requester_is_false_should_keep_unrestricted_default():
    # Regression: the default (False) must behave exactly as before this
    # change -- omitting requester_user_id stays a valid, supported call.
    client = ScriptedClient([_submit("ESCALATION_REQUIRED")])
    agent = ResolverAgent(client=client)  # require_verified_requester defaults to False

    result = agent.resolve("Hi, I have a problem.")

    assert result["action_taken"]["decision"] == "ESCALATION_REQUIRED"
