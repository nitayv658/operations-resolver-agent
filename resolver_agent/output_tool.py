"""The structured-output contract for a resolved case.

Instead of asking the model to free-type JSON at the end and parsing it with
regex, the required output shape is exposed *as a tool*: ``submit_resolution``.
The model calling it is an ordinary tool_use turn, so its arguments arrive
already schema-validated by the API before this code ever sees them -- no
regex, no "hope it's valid JSON".
"""

from __future__ import annotations

from typing import Any, Dict, List

from .tool_loop import ToolCallRecord

SUBMIT_RESOLUTION_TOOL_NAME = "submit_resolution"

DECISION_VALUES = [
    "AUTO_REFUND_APPROVED",
    "REJECTED",
    "ESCALATION_REQUIRED",
    "CANNOT_RESOLVE",
]

SUBMIT_RESOLUTION_SCHEMA: Dict[str, Any] = {
    "name": SUBMIT_RESOLUTION_TOOL_NAME,
    "description": (
        "Record the final resolution of this support case and end it. Call "
        "this exactly once, as your last action, only after you have looked "
        "up whatever order/user/policy data you need and reached a decision. "
        "Do not call any other tool after this one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning_chain": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Ordered list of concrete facts the decision is based on -- "
                    "quote real values and policy ids actually seen in tool "
                    "results (order id, amounts, dates, tier, policy ids). Do "
                    "not write generic statements that could apply to any "
                    "ticket."
                ),
            },
            "action_taken": {
                "type": "object",
                "properties": {
                    "tools_called": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of every GlobalCart tool called, in order.",
                    },
                    "decision": {
                        "type": "string",
                        "enum": DECISION_VALUES,
                        "description": (
                            "AUTO_REFUND_APPROVED: process_refund returned "
                            "APPROVED. REJECTED: the claim is not eligible "
                            "(policy says no). ESCALATION_REQUIRED: "
                            "process_refund returned ESCALATION_REQUIRED, or "
                            "the case needs a human for any other reason. "
                            "CANNOT_RESOLVE: the order/user needed to decide "
                            "could not be found."
                        ),
                    },
                    "refund_amount": {
                        "type": ["number", "null"],
                        "description": "process_refund's approved_amount, or null if none was approved.",
                    },
                    "refund_id": {
                        "type": ["string", "null"],
                        "description": "process_refund's refund_id, or null if none was approved.",
                    },
                },
                "required": ["tools_called", "decision"],
            },
            "customer_response": {
                "type": "string",
                "description": (
                    "The reply to send the customer. Must match `decision` "
                    "exactly -- never describe a refund as done unless "
                    "decision is AUTO_REFUND_APPROVED. Match the customer's "
                    "tone and language."
                ),
            },
        },
        "required": ["reasoning_chain", "action_taken", "customer_response"],
    },
}


def validate_resolution(
    resolution: Dict[str, Any], tool_calls: List[ToolCallRecord]
) -> List[str]:
    """Cross-check the stated decision against what the tools actually returned.

    Returns a list of human-readable violations (empty if consistent). A
    system prompt telling the model "match the decision to the tool result"
    is a suggestion it can still get wrong under unusual phrasing; this makes
    the requirement mechanical -- the same "guardrail in code, not prompt"
    philosophy ``process_refund`` itself uses for the refund cap.
    """
    warnings: List[str] = []
    action = resolution.get("action_taken") or {}
    decision = action.get("decision")

    refund_calls = [
        c for c in tool_calls if c.name == "process_refund" and isinstance(c.result, dict)
    ]
    last_refund = refund_calls[-1].result if refund_calls else None

    error_calls = [
        c
        for c in tool_calls
        if isinstance(c.result, dict) and "error" in c.result and c.name != SUBMIT_RESOLUTION_TOOL_NAME
    ]

    if last_refund is not None:
        status = last_refund.get("status")

        if status == "ESCALATION_REQUIRED" and decision != "ESCALATION_REQUIRED":
            warnings.append(
                f"process_refund returned ESCALATION_REQUIRED but decision was '{decision}'."
            )
        if status == "REJECTED" and decision not in ("REJECTED", "ESCALATION_REQUIRED"):
            warnings.append(f"process_refund returned REJECTED but decision was '{decision}'.")
        if status != "APPROVED" and decision == "AUTO_REFUND_APPROVED":
            warnings.append(
                f"decision is AUTO_REFUND_APPROVED but process_refund's last status was '{status}'."
            )
        if status == "APPROVED":
            if decision != "AUTO_REFUND_APPROVED":
                warnings.append(
                    f"process_refund returned APPROVED but decision was '{decision}'."
                )
            approved_amount = last_refund.get("approved_amount")
            if action.get("refund_amount") != approved_amount:
                warnings.append(
                    f"customer-facing refund_amount ({action.get('refund_amount')}) does not "
                    f"match process_refund's approved_amount ({approved_amount})."
                )
            if action.get("refund_id") != last_refund.get("refund_id"):
                warnings.append(
                    f"customer-facing refund_id ({action.get('refund_id')}) does not match "
                    f"process_refund's refund_id ({last_refund.get('refund_id')})."
                )

            requested = last_refund.get("requested_amount")
            cap = last_refund.get("auto_refund_cap_usd")
            order_id = last_refund.get("order_id")
            order_lookup = next(
                (
                    c.result.get("total_amount")
                    for c in tool_calls
                    if c.name == "get_order_details"
                    and isinstance(c.result, dict)
                    and c.result.get("order_id") == order_id
                ),
                None,
            )
            if (
                requested is not None
                and cap is not None
                and order_lookup is not None
                and requested < order_lookup
                and requested == cap
            ):
                warnings.append(
                    f"process_refund was called with amount={requested} == the auto-refund cap "
                    f"({cap}), below the order total ({order_lookup}) -- looks like the agent "
                    "under-requested the refund specifically to avoid triggering "
                    "ESCALATION_REQUIRED, instead of requesting the true amount owed."
                )

    if last_refund is None and decision == "AUTO_REFUND_APPROVED":
        warnings.append("decision is AUTO_REFUND_APPROVED but process_refund was never called.")

    if (
        error_calls
        and last_refund is None
        and decision not in ("CANNOT_RESOLVE", "ESCALATION_REQUIRED", "REJECTED")
    ):
        codes = ", ".join(sorted({c.result.get("error") for c in error_calls}))
        warnings.append(
            f"a tool returned an error ({codes}) and no refund was processed, "
            f"but decision was '{decision}'."
        )

    return warnings
