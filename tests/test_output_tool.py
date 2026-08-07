"""Tests for output_tool.validate_resolution(), validate_schema() and
enforce_resolution().

Wherever practical, tool results are produced by calling the real
starter-kit functions (via the `gc` module) against real fixture orders,
not hand-written mocks -- so these tests exercise the exact dict shapes
mock_services.py actually returns. Only two boundary cases (which the
fixture data doesn't happen to land on exactly) use hand-built
ToolCallRecords, and those are noted inline.
"""

from __future__ import annotations

from resolver_agent.agent import gc
from resolver_agent.output_tool import DECISION_VALUES, enforce_resolution, validate_resolution, validate_schema
from resolver_agent.tool_loop import ToolCallRecord


def _order_call(order_id: str) -> ToolCallRecord:
    return ToolCallRecord("get_order_details", {"order_id": order_id}, gc.get_order_details(order_id))


def _refund_call(order_id: str, amount: float) -> ToolCallRecord:
    return ToolCallRecord(
        "process_refund",
        {"order_id": order_id, "amount": amount},
        gc.process_refund(order_id, amount),
    )


def test_validate_resolution_when_decision_matches_approval_should_return_no_warnings():
    # ORD-1001: VIP, $35, damaged -- real flow approves in full.
    order_call = _order_call("ORD-1001")
    refund_call = _refund_call("ORD-1001", order_call.result["total_amount"])
    assert refund_call.result["status"] == "APPROVED"

    resolution = {
        "action_taken": {
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": refund_call.result["approved_amount"],
            "refund_id": refund_call.result["refund_id"],
        }
    }
    assert validate_resolution(resolution, [order_call, refund_call]) == []


def test_validate_resolution_when_escalation_required_but_decision_claims_approved_should_warn():
    # ORD-1002: $150, above the $50 Standard cap -- real process_refund escalates.
    order_call = _order_call("ORD-1002")
    refund_call = _refund_call("ORD-1002", order_call.result["total_amount"])
    assert refund_call.result["status"] == "ESCALATION_REQUIRED"

    resolution = {
        "action_taken": {
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": 150.0,
            "refund_id": "RF-FAKE",
        }
    }
    warnings = validate_resolution(resolution, [order_call, refund_call])
    assert any("ESCALATION_REQUIRED but decision was" in w for w in warnings)


def test_validate_resolution_when_rejected_but_decision_claims_approved_should_warn():
    # ORD-1003: return requested 60 days after delivery -- outside the window,
    # so the real check_return_policy/process_refund path rejects it.
    order_call = _order_call("ORD-1003")
    refund_call = _refund_call("ORD-1003", order_call.result["total_amount"])
    assert refund_call.result["status"] == "REJECTED"

    resolution = {
        "action_taken": {
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": order_call.result["total_amount"],
            "refund_id": "RF-FAKE",
        }
    }
    warnings = validate_resolution(resolution, [order_call, refund_call])
    assert any("REJECTED but decision was" in w for w in warnings)


def test_validate_resolution_when_customer_facing_amount_mismatches_approved_amount_should_warn():
    order_call = _order_call("ORD-1001")
    refund_call = _refund_call("ORD-1001", order_call.result["total_amount"])

    resolution = {
        "action_taken": {
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": 999.0,  # does not match refund_call.result["approved_amount"]
            "refund_id": refund_call.result["refund_id"],
        }
    }
    warnings = validate_resolution(resolution, [order_call, refund_call])
    assert any("refund_amount" in w and "does not match" in w for w in warnings)


def test_validate_resolution_when_customer_facing_refund_id_mismatches_should_warn():
    order_call = _order_call("ORD-1001")
    refund_call = _refund_call("ORD-1001", order_call.result["total_amount"])

    resolution = {
        "action_taken": {
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": refund_call.result["approved_amount"],
            "refund_id": "RF-MADE-UP",
        }
    }
    warnings = validate_resolution(resolution, [order_call, refund_call])
    assert any("refund_id" in w and "does not match" in w for w in warnings)


def test_validate_resolution_when_agent_under_requests_to_dodge_cap_should_warn():
    # ORD-1011: $52, Standard cap $50 -- the exact live bug: requesting only
    # $50 (the cap) instead of the true $52 owed gets APPROVED, dodging the
    # ESCALATION_REQUIRED that requesting the real amount would trigger.
    order_call = _order_call("ORD-1011")
    assert order_call.result["total_amount"] == 52.0
    refund_call = _refund_call("ORD-1011", 50.0)
    assert refund_call.result["status"] == "APPROVED"  # confirms the dodge "works" at the tool level

    resolution = {
        "action_taken": {
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": 50.0,
            "refund_id": refund_call.result["refund_id"],
        }
    }
    warnings = validate_resolution(resolution, [order_call, refund_call])
    assert any("under-requested the refund" in w for w in warnings)


def test_validate_resolution_when_requested_amount_equals_order_total_and_cap_should_not_false_positive():
    # Boundary the real fixtures don't happen to land on: an order whose full
    # total *is* exactly the cap. requested < order_total must be false here,
    # so the under-request warning must not fire even though requested == cap.
    order_call = ToolCallRecord(
        "get_order_details", {"order_id": "ORD-9001"}, {"order_id": "ORD-9001", "total_amount": 50.0}
    )
    refund_call = ToolCallRecord(
        "process_refund",
        {"order_id": "ORD-9001", "amount": 50.0},
        {
            "order_id": "ORD-9001",
            "status": "APPROVED",
            "requested_amount": 50.0,
            "approved_amount": 50.0,
            "auto_refund_cap_usd": 50.0,
            "refund_id": "RF-9001-5000",
        },
    )
    resolution = {
        "action_taken": {
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": 50.0,
            "refund_id": "RF-9001-5000",
        }
    }
    assert validate_resolution(resolution, [order_call, refund_call]) == []


def test_validate_resolution_when_approved_decision_without_process_refund_call_should_warn():
    order_call = _order_call("ORD-1001")
    resolution = {
        "action_taken": {"decision": "AUTO_REFUND_APPROVED", "refund_amount": 35.0, "refund_id": "RF-MADE-UP"}
    }
    warnings = validate_resolution(resolution, [order_call])
    assert any("process_refund was never called" in w for w in warnings)


def test_validate_resolution_when_tool_error_and_decision_cannot_resolve_should_not_warn():
    # ORD-9999 does not exist -- the hallucination-trap shape.
    error_call = ToolCallRecord("get_order_details", {"order_id": "ORD-9999"}, gc.get_order_details("ORD-9999"))
    assert "error" in error_call.result

    resolution = {
        "action_taken": {"decision": "CANNOT_RESOLVE", "refund_amount": None, "refund_id": None}
    }
    assert validate_resolution(resolution, [error_call]) == []


def test_validate_resolution_when_tool_error_but_decision_claims_approved_should_warn():
    error_call = ToolCallRecord("get_order_details", {"order_id": "ORD-9999"}, gc.get_order_details("ORD-9999"))

    resolution = {
        "action_taken": {"decision": "AUTO_REFUND_APPROVED", "refund_amount": 300.0, "refund_id": "RF-MADE-UP"}
    }
    warnings = validate_resolution(resolution, [error_call])
    assert any("a tool returned an error" in w for w in warnings)


# --------------------------------------------------------------------------- #
# validate_schema() -- regression tests for the two gaps proven during the
# principal-level architecture review: an out-of-enum decision and a
# completely empty resolution both used to pass through with zero warnings.
# --------------------------------------------------------------------------- #


def test_validate_schema_when_resolution_is_completely_empty_should_report_every_missing_field():
    # The exact review repro: validate_resolution({}, []) returned [] even
    # though nothing required is present.
    errors = validate_schema({})
    assert any("reasoning_chain" in e for e in errors)
    assert any("action_taken" in e for e in errors)
    assert any("customer_response" in e for e in errors)


def test_validate_schema_when_decision_is_out_of_enum_should_report_it():
    # The exact review repro: "MAYBE_REFUND_LATER" is not one of
    # DECISION_VALUES, and nothing previously caught that.
    resolution = {
        "reasoning_chain": ["..."],
        "action_taken": {"tools_called": [], "decision": "MAYBE_REFUND_LATER"},
        "customer_response": "...",
    }
    errors = validate_schema(resolution)
    assert any("decision" in e and "MAYBE_REFUND_LATER" in e for e in errors)


def test_validate_schema_when_customer_response_is_missing_should_report_it():
    resolution = {
        "reasoning_chain": ["..."],
        "action_taken": {"tools_called": [], "decision": "ESCALATION_REQUIRED"},
    }
    errors = validate_schema(resolution)
    assert any("customer_response" in e for e in errors)


def test_validate_schema_when_customer_response_is_blank_should_report_it():
    resolution = {
        "reasoning_chain": ["..."],
        "action_taken": {"tools_called": [], "decision": "ESCALATION_REQUIRED"},
        "customer_response": "   ",
    }
    errors = validate_schema(resolution)
    assert any("customer_response" in e for e in errors)


def test_validate_schema_when_resolution_is_well_formed_should_return_no_errors():
    resolution = {
        "reasoning_chain": ["Order ORD-1001 delivered, $35.00, damaged_on_arrival."],
        "action_taken": {
            "tools_called": ["get_order_details", "process_refund"],
            "decision": "AUTO_REFUND_APPROVED",
            "refund_amount": 35.0,
            "refund_id": "RF-1001-3500",
        },
        "customer_response": "Your refund has been approved.",
    }
    assert validate_schema(resolution) == []


def test_validate_schema_accepts_every_real_decision_value():
    for decision in DECISION_VALUES:
        resolution = {
            "reasoning_chain": ["..."],
            "action_taken": {"tools_called": [], "decision": decision},
            "customer_response": "...",
        }
        assert validate_schema(resolution) == []


# --------------------------------------------------------------------------- #
# enforce_resolution() -- detection AND deterministic correction. One case
# per override rule, plus the soft-correction-only path and the clean
# happy path.
# --------------------------------------------------------------------------- #


def _resolution(decision, refund_amount=None, refund_id=None, customer_response="..."):
    return {
        "reasoning_chain": ["..."],
        "action_taken": {
            "tools_called": [],
            "decision": decision,
            "refund_amount": refund_amount,
            "refund_id": refund_id,
        },
        "customer_response": customer_response,
    }


def test_enforce_resolution_when_no_issues_found_should_return_resolution_unchanged():
    order_call = _order_call("ORD-1001")
    refund_call = _refund_call("ORD-1001", order_call.result["total_amount"])
    resolution = _resolution(
        "AUTO_REFUND_APPROVED",
        refund_amount=refund_call.result["approved_amount"],
        refund_id=refund_call.result["refund_id"],
        customer_response="Your $35.00 refund has been approved.",
    )

    corrected, warnings, corrections = enforce_resolution(resolution, [order_call, refund_call])

    assert warnings == []
    assert corrections == []
    assert corrected == resolution


def test_enforce_resolution_when_escalation_required_but_claimed_approved_should_override_to_escalate():
    # ORD-1002: $150, above the Standard $50 cap.
    order_call = _order_call("ORD-1002")
    refund_call = _refund_call("ORD-1002", order_call.result["total_amount"])
    assert refund_call.result["status"] == "ESCALATION_REQUIRED"

    resolution = _resolution("AUTO_REFUND_APPROVED", refund_amount=150.0, refund_id="RF-FAKE")
    corrected, warnings, corrections = enforce_resolution(resolution, [order_call, refund_call])

    assert corrected["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert corrected["action_taken"]["refund_amount"] is None
    assert corrected["action_taken"]["refund_id"] is None
    assert "escalated" in corrected["customer_response"].lower()
    assert warnings  # the original inconsistency is still reported
    assert corrections
    assert "AUTO_REFUND_APPROVED" in corrections[0] and "ESCALATION_REQUIRED" in corrections[0]


def test_enforce_resolution_when_rejected_but_claimed_approved_should_override_to_rejected():
    # ORD-1003: outside the return window.
    order_call = _order_call("ORD-1003")
    refund_call = _refund_call("ORD-1003", order_call.result["total_amount"])
    assert refund_call.result["status"] == "REJECTED"

    resolution = _resolution("AUTO_REFUND_APPROVED", refund_amount=order_call.result["total_amount"])
    corrected, _warnings, corrections = enforce_resolution(resolution, [order_call, refund_call])

    assert corrected["action_taken"]["decision"] == "REJECTED"
    assert corrected["action_taken"]["refund_amount"] is None
    assert corrections


def test_enforce_resolution_when_actually_approved_but_claimed_rejected_should_override_to_approved():
    # ORD-1001: VIP, $35, damaged -- genuinely approved. The model
    # under-reports it as rejected; enforcement should correct decision
    # AND fill in the real amount/id from the tool, not just flip a label.
    order_call = _order_call("ORD-1001")
    refund_call = _refund_call("ORD-1001", order_call.result["total_amount"])
    assert refund_call.result["status"] == "APPROVED"

    resolution = _resolution("REJECTED")
    corrected, _warnings, corrections = enforce_resolution(resolution, [order_call, refund_call])

    assert corrected["action_taken"]["decision"] == "AUTO_REFUND_APPROVED"
    assert corrected["action_taken"]["refund_amount"] == refund_call.result["approved_amount"]
    assert corrected["action_taken"]["refund_id"] == refund_call.result["refund_id"]
    assert corrections


def test_enforce_resolution_when_under_request_dodge_detected_should_override_to_escalate():
    # ORD-1011: $52, Standard cap $50 -- the exact live-reproduced bug.
    # process_refund itself returned APPROVED (for the gamed $50 request),
    # but enforcement must still escalate rather than trust that approval.
    order_call = _order_call("ORD-1011")
    refund_call = _refund_call("ORD-1011", 50.0)
    assert refund_call.result["status"] == "APPROVED"

    resolution = _resolution("AUTO_REFUND_APPROVED", refund_amount=50.0, refund_id=refund_call.result["refund_id"])
    corrected, warnings, corrections = enforce_resolution(resolution, [order_call, refund_call])

    assert corrected["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert corrected["action_taken"]["refund_amount"] is None
    assert any("under-requested the refund" in w for w in warnings)
    assert corrections


def test_enforce_resolution_when_approved_claimed_without_any_process_refund_call_should_override_to_escalate():
    order_call = _order_call("ORD-1001")
    resolution = _resolution("AUTO_REFUND_APPROVED", refund_amount=35.0, refund_id="RF-MADE-UP")

    corrected, _warnings, corrections = enforce_resolution(resolution, [order_call])

    assert corrected["action_taken"]["decision"] == "ESCALATION_REQUIRED"
    assert corrected["action_taken"]["refund_amount"] is None
    assert corrections


def test_enforce_resolution_when_tool_errored_but_claimed_approved_should_override_to_cannot_resolve():
    # ORD-9999 does not exist -- the hallucination-trap shape.
    error_call = ToolCallRecord("get_order_details", {"order_id": "ORD-9999"}, gc.get_order_details("ORD-9999"))
    resolution = _resolution("AUTO_REFUND_APPROVED", refund_amount=300.0, refund_id="RF-MADE-UP")

    corrected, _warnings, corrections = enforce_resolution(resolution, [error_call])

    assert corrected["action_taken"]["decision"] == "CANNOT_RESOLVE"
    assert corrected["action_taken"]["refund_amount"] is None
    assert corrections


def test_enforce_resolution_when_only_refund_amount_is_wrong_should_correct_only_that_field():
    # Decision type is already correct (APPROVED matches APPROVED) -- this
    # is a soft correction, not a full decision override.
    order_call = _order_call("ORD-1001")
    refund_call = _refund_call("ORD-1001", order_call.result["total_amount"])

    resolution = _resolution(
        "AUTO_REFUND_APPROVED", refund_amount=999.0, refund_id=refund_call.result["refund_id"]
    )
    corrected, warnings, corrections = enforce_resolution(resolution, [order_call, refund_call])

    assert corrected["action_taken"]["decision"] == "AUTO_REFUND_APPROVED"  # unchanged
    assert corrected["action_taken"]["refund_amount"] == refund_call.result["approved_amount"]
    assert corrected["action_taken"]["refund_id"] == refund_call.result["refund_id"]  # untouched, was already correct
    assert any("refund_amount" in w for w in warnings)
    assert corrections
    # customer-facing text must reflect the corrected amount, not the wrong one
    assert "999" not in corrected["customer_response"]
