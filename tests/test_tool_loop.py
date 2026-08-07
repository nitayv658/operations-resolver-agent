"""Tests for tool_loop.run_tool_loop().

Only the model is faked (via ScriptedClient, see tests/helpers.py) -- real
tool dispatch runs through the real starter-kit mock_services functions, so
these tests prove the loop's own mechanics: dispatch, the repeat-call guard,
both max_iterations termination paths, unknown-tool handling, and
programmer-error wrapping.
"""

from __future__ import annotations

import logging

import anthropic
import pytest

from resolver_agent.output_tool import SUBMIT_RESOLUTION_TOOL_NAME
from resolver_agent.tool_loop import ModelAPIError, ToolExecutionError, run_tool_loop

from .helpers import (
    ScriptedClient,
    ScriptedResponse,
    fake_api_connection_error,
    fake_bad_request_error,
    text_block,
    tool_use_block,
)


def _run(script, tool_schemas, tool_registry, max_iterations=8, log_context=None):
    client = ScriptedClient(script)
    messages = [{"role": "user", "content": "(scripted ticket)"}]
    return run_tool_loop(
        client=client,
        model="mock",
        system="(unused)",
        messages=messages,
        tool_schemas=tool_schemas,
        tool_registry=tool_registry,
        stop_tool_name=SUBMIT_RESOLUTION_TOOL_NAME,
        max_iterations=max_iterations,
        log_context=log_context,
    )


def test_run_tool_loop_when_model_calls_real_tools_then_stops_should_dispatch_them_via_registry(
    tool_schemas, tool_registry
):
    result = _run(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("get_user_profile", {"user_id": "USR-101"})]),
            ScriptedResponse([tool_use_block("check_return_policy", {"order_id": "ORD-1001", "reason": "damaged_on_arrival"})]),
            ScriptedResponse([tool_use_block("process_refund", {"order_id": "ORD-1001", "amount": 35.0, "reason": "damaged_on_arrival"})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RESOLUTION_TOOL_NAME,
                        {
                            "reasoning_chain": ["..."],
                            "action_taken": {"tools_called": [], "decision": "AUTO_REFUND_APPROVED", "refund_amount": 35.0, "refund_id": "RF-1001-3500"},
                            "customer_response": "...",
                        },
                    )
                ]
            ),
        ],
        tool_schemas,
        tool_registry,
    )

    assert result.stopped_reason == "stop"
    called_names = [c.name for c in result.tool_calls]
    assert called_names == [
        "get_order_details",
        "get_user_profile",
        "check_return_policy",
        "process_refund",
        SUBMIT_RESOLUTION_TOOL_NAME,
    ]
    process_refund_call = next(c for c in result.tool_calls if c.name == "process_refund")
    assert process_refund_call.result["status"] == "APPROVED"  # real dispatch, real fixture data


def test_run_tool_loop_when_same_call_repeated_should_execute_only_once(tool_schemas, tool_registry):
    result = _run(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),  # exact repeat
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RESOLUTION_TOOL_NAME,
                        {
                            "reasoning_chain": ["gave up after repeat"],
                            "action_taken": {"tools_called": [], "decision": "ESCALATION_REQUIRED", "refund_amount": None, "refund_id": None},
                            "customer_response": "...",
                        },
                    )
                ]
            ),
        ],
        tool_schemas,
        tool_registry,
    )

    executed = [c for c in result.tool_calls if c.name == "get_order_details"]
    assert len(executed) == 1  # not 2 -- the repeat was refused, not re-run

    refusal_messages = [
        block
        for m in result.messages
        if m.get("role") == "user" and isinstance(m.get("content"), list)
        for block in m["content"]
        if isinstance(block, dict) and "already called this tool" in str(block.get("content", ""))
    ]
    assert len(refusal_messages) == 1


def test_run_tool_loop_when_model_never_stops_should_force_final_call_and_report_max_iterations(
    tool_schemas, tool_registry
):
    # Distinct order ids each round -- the dedup guard would otherwise refuse
    # a literal repeat and the loop would never advance to the forced call.
    result = _run(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1002"})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RESOLUTION_TOOL_NAME,
                        {
                            "reasoning_chain": ["forced to conclude"],
                            "action_taken": {"tools_called": [], "decision": "ESCALATION_REQUIRED", "refund_amount": None, "refund_id": None},
                            "customer_response": "Escalating.",
                        },
                    )
                ]
            ),
        ],
        tool_schemas,
        tool_registry,
        max_iterations=2,
    )

    assert result.stopped_reason == "max_iterations"
    submit_call = next(c for c in result.tool_calls if c.name == SUBMIT_RESOLUTION_TOOL_NAME)
    assert submit_call.input["action_taken"]["decision"] == "ESCALATION_REQUIRED"


def test_run_tool_loop_when_model_ignores_forced_stop_should_still_terminate_without_raising(
    tool_schemas, tool_registry
):
    result = _run(
        [
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
            ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1002"})]),
            # forced call: model still doesn't comply, just talks
            ScriptedResponse([text_block("I'm not sure what to do.")], stop_reason="end_turn"),
        ],
        tool_schemas,
        tool_registry,
        max_iterations=2,
    )

    assert result.stopped_reason == "max_iterations"
    assert not any(c.name == SUBMIT_RESOLUTION_TOOL_NAME for c in result.tool_calls)


def test_run_tool_loop_when_model_names_unknown_tool_should_return_error_dict_without_crashing(
    tool_schemas, tool_registry
):
    result = _run(
        [
            ScriptedResponse([tool_use_block("delete_customer_account", {"user_id": "USR-101"})]),
            ScriptedResponse(
                [
                    tool_use_block(
                        SUBMIT_RESOLUTION_TOOL_NAME,
                        {
                            "reasoning_chain": ["no such tool"],
                            "action_taken": {"tools_called": [], "decision": "CANNOT_RESOLVE", "refund_amount": None, "refund_id": None},
                            "customer_response": "...",
                        },
                    )
                ]
            ),
        ],
        tool_schemas,
        tool_registry,
    )

    unknown_call = next(c for c in result.tool_calls if c.name == "delete_customer_account")
    assert unknown_call.result == {
        "error": "UNKNOWN_TOOL",
        "message": "No such tool: delete_customer_account",
    }


def test_run_tool_loop_when_tool_raises_type_error_should_wrap_as_tool_execution_error(
    tool_schemas, tool_registry
):
    # get_order_details raises TypeError for a non-string order_id -- a
    # genuine programmer/schema-violation error, distinct from the
    # business-error dicts mock_services returns for bad *values*.
    with pytest.raises(ToolExecutionError):
        _run(
            [ScriptedResponse([tool_use_block("get_order_details", {"order_id": 12345})])],
            tool_schemas,
            tool_registry,
        )


def test_run_tool_loop_when_api_call_raises_should_wrap_as_model_api_error_with_partial_trace(
    tool_schemas, tool_registry
):
    # First round succeeds and executes a real tool call; the second round's
    # API call fails the way the SDK actually fails after its own internal
    # retries are exhausted (or immediately, for a non-retryable error like
    # this one). The partial trace from round 1 must not be lost.
    with pytest.raises(ModelAPIError) as exc_info:
        _run(
            [
                ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
                fake_api_connection_error(),
            ],
            tool_schemas,
            tool_registry,
        )

    err = exc_info.value
    assert len(err.tool_calls) == 1
    assert err.tool_calls[0].name == "get_order_details"
    assert isinstance(err.__cause__, anthropic.APIConnectionError)


def test_run_tool_loop_when_api_call_raises_on_the_very_first_call_should_still_wrap_cleanly(
    tool_schemas, tool_registry
):
    with pytest.raises(ModelAPIError) as exc_info:
        _run([fake_bad_request_error()], tool_schemas, tool_registry)

    err = exc_info.value
    assert err.tool_calls == []
    assert isinstance(err.__cause__, anthropic.BadRequestError)


def _submit(decision="ESCALATION_REQUIRED"):
    return ScriptedResponse(
        [
            tool_use_block(
                SUBMIT_RESOLUTION_TOOL_NAME,
                {
                    "reasoning_chain": ["..."],
                    "action_taken": {"tools_called": [], "decision": decision, "refund_amount": None, "refund_id": None},
                    "customer_response": "...",
                },
            )
        ]
    )


def test_run_tool_loop_when_repeat_call_refused_should_log_warning(tool_schemas, tool_registry, caplog):
    with caplog.at_level(logging.WARNING, logger="resolver_agent.tool_loop"):
        _run(
            [
                ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
                ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),  # repeat
                _submit(),
            ],
            tool_schemas,
            tool_registry,
        )

    records = [r for r in caplog.records if r.getMessage() == "tool_loop.repeat_call_refused"]
    assert len(records) == 1
    assert records[0].fields["tool"] == "get_order_details"


def test_run_tool_loop_when_unknown_tool_requested_should_log_warning(tool_schemas, tool_registry, caplog):
    with caplog.at_level(logging.WARNING, logger="resolver_agent.tool_loop"):
        _run(
            [
                ScriptedResponse([tool_use_block("delete_customer_account", {"user_id": "USR-101"})]),
                _submit("CANNOT_RESOLVE"),
            ],
            tool_schemas,
            tool_registry,
        )

    records = [r for r in caplog.records if r.getMessage() == "tool_loop.unknown_tool_requested"]
    assert len(records) == 1
    assert records[0].fields["tool"] == "delete_customer_account"


def test_run_tool_loop_when_max_iterations_reached_should_log_warning(tool_schemas, tool_registry, caplog):
    with caplog.at_level(logging.WARNING, logger="resolver_agent.tool_loop"):
        _run(
            [
                ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
                ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1002"})]),
                _submit(),
            ],
            tool_schemas,
            tool_registry,
            max_iterations=2,
        )

    assert any(r.getMessage() == "tool_loop.max_iterations_reached" for r in caplog.records)


def test_run_tool_loop_when_log_context_given_should_be_merged_into_every_record(
    tool_schemas, tool_registry, caplog
):
    with caplog.at_level(logging.DEBUG, logger="resolver_agent.tool_loop"):
        _run(
            [
                ScriptedResponse([tool_use_block("get_order_details", {"order_id": "ORD-1001"})]),
                _submit(),
            ],
            tool_schemas,
            tool_registry,
            log_context={"case_id": "test-case-42"},
        )

    tool_loop_records = [r for r in caplog.records if r.name == "resolver_agent.tool_loop"]
    assert tool_loop_records  # sanity: at least the tool_executed DEBUG record fired
    assert all(r.fields.get("case_id") == "test-case-42" for r in tool_loop_records)
