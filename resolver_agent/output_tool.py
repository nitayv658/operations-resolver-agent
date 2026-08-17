"""The structured-output contract for a resolved case.

Instead of asking the model to free-type JSON at the end and parsing it with
regex, the required output shape is exposed *as a tool*: ``submit_resolution``.
The model calling it is an ordinary tool_use turn, so its arguments arrive
already schema-validated by the API before this code ever sees them -- no
regex, no "hope it's valid JSON".

That API-side schema constraint is real but is not a substitute for
independent verification in our own code -- an out-of-enum ``decision`` or a
missing required field would still reach us if the model ever produced one.
Three layers live in this file:

1. :func:`validate_schema` -- structural/type completeness, no tool_calls
   needed. Catches malformed output before anything else touches it.
2. :func:`validate_resolution` -- cross-checks a (structurally valid)
   resolution's stated decision against what the tools actually returned.
   Detection only, kept as its own function for backward compatibility.
3. :func:`enforce_resolution` -- uses the same findings as (2) but *acts* on
   them: deterministically overrides the decision/response to match the
   tools' ground truth rather than merely flagging the inconsistency. No
   second LLM call, no policy math reimplemented here -- every override is
   either a direct read of a tool's own ``status`` field or a conservative
   "escalate/cannot-resolve when uncertain" default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .tool_loop import ToolCallRecord

SUBMIT_RESOLUTION_TOOL_NAME = "submit_resolution"

DECISION_VALUES = [
    "AUTO_REFUND_APPROVED",
    "REJECTED",
    "ESCALATION_REQUIRED",
    "CANNOT_RESOLVE",
]

# Most-conservative-wins tie-break for when more than one hard override
# fires for the same resolution -- this does happen (e.g. process_refund
# returning ESCALATION_REQUIRED while the model claims AUTO_REFUND_APPROVED
# trips both the direct ESCALATION_REQUIRED check and the "status != APPROVED
# but decision == AUTO_REFUND_APPROVED" check below), the two findings just
# always agree on the same override_decision for a given tool status, so the
# tie-break never actually changes the outcome -- it exists to keep the
# result deterministic rather than to resolve a real disagreement.
_DECISION_PRIORITY = {
    "CANNOT_RESOLVE": 0,
    "ESCALATION_REQUIRED": 1,
    "REJECTED": 2,
    "AUTO_REFUND_APPROVED": 3,
}

# Sentinel distinguishing "this finding doesn't touch this field" from "the
# correct value for this field is None" -- refund_amount/refund_id are both
# legitimately nullable, so None can't double as "leave it alone".
_UNSET = object()

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


def validate_schema(resolution: Dict[str, Any]) -> List[str]:
    """Check that a resolution has the required shape, independent of the
    Anthropic API's own tool-schema enforcement.

    Returns a list of human-readable errors (empty if the shape is valid).
    This exists because the API's JSON-schema constraint on the model's tool
    call is a real guarantee but not one this code should simply trust --
    a missing field or an out-of-enum ``decision`` should be caught here,
    not silently accepted.
    """
    errors: List[str] = []

    reasoning_chain = resolution.get("reasoning_chain")
    if not isinstance(reasoning_chain, list) or not all(isinstance(item, str) for item in reasoning_chain):
        errors.append("reasoning_chain is missing or is not a list of strings.")

    action = resolution.get("action_taken")
    if not isinstance(action, dict):
        errors.append("action_taken is missing or is not an object.")
    else:
        if not isinstance(action.get("tools_called"), list):
            errors.append("action_taken.tools_called is missing or is not a list.")
        if action.get("decision") not in DECISION_VALUES:
            errors.append(
                f"action_taken.decision is missing or not one of {DECISION_VALUES}: "
                f"got {action.get('decision')!r}."
            )

    customer_response = resolution.get("customer_response")
    if not isinstance(customer_response, str) or not customer_response.strip():
        errors.append("customer_response is missing, not a string, or empty.")

    return errors


@dataclass
class _Finding:
    message: str
    override_decision: Optional[str] = None
    override_refund_amount: Any = _UNSET
    override_refund_id: Any = _UNSET


def _find_issues(resolution: Dict[str, Any], tool_calls: List[ToolCallRecord]) -> List[_Finding]:
    """Cross-check a resolution's stated decision against what the tools
    actually returned. Shared by :func:`validate_resolution` (message
    strings only) and :func:`enforce_resolution` (messages + the correction
    each finding implies).
    """
    findings: List[_Finding] = []
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
            findings.append(
                _Finding(
                    f"process_refund returned ESCALATION_REQUIRED but decision was '{decision}'.",
                    override_decision="ESCALATION_REQUIRED",
                    override_refund_amount=None,
                    override_refund_id=None,
                )
            )
        if status == "REJECTED" and decision not in ("REJECTED", "ESCALATION_REQUIRED"):
            findings.append(
                _Finding(
                    f"process_refund returned REJECTED but decision was '{decision}'.",
                    override_decision="REJECTED",
                    override_refund_amount=None,
                    override_refund_id=None,
                )
            )
        if status != "APPROVED" and decision == "AUTO_REFUND_APPROVED":
            mapped = status if status in ("ESCALATION_REQUIRED", "REJECTED") else "ESCALATION_REQUIRED"
            findings.append(
                _Finding(
                    f"decision is AUTO_REFUND_APPROVED but process_refund's last status was '{status}'.",
                    override_decision=mapped,
                    override_refund_amount=None,
                    override_refund_id=None,
                )
            )
        if status == "APPROVED":
            if decision != "AUTO_REFUND_APPROVED":
                findings.append(
                    _Finding(
                        f"process_refund returned APPROVED but decision was '{decision}'.",
                        override_decision="AUTO_REFUND_APPROVED",
                        override_refund_amount=last_refund.get("approved_amount"),
                        override_refund_id=last_refund.get("refund_id"),
                    )
                )
            approved_amount = last_refund.get("approved_amount")
            if action.get("refund_amount") != approved_amount:
                findings.append(
                    _Finding(
                        f"customer-facing refund_amount ({action.get('refund_amount')}) does not "
                        f"match process_refund's approved_amount ({approved_amount}).",
                        override_refund_amount=approved_amount,
                    )
                )
            if action.get("refund_id") != last_refund.get("refund_id"):
                findings.append(
                    _Finding(
                        f"customer-facing refund_id ({action.get('refund_id')}) does not match "
                        f"process_refund's refund_id ({last_refund.get('refund_id')}).",
                        override_refund_id=last_refund.get("refund_id"),
                    )
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
                findings.append(
                    _Finding(
                        f"process_refund was called with amount={requested} == the auto-refund cap "
                        f"({cap}), below the order total ({order_lookup}) -- looks like the agent "
                        "under-requested the refund specifically to avoid triggering "
                        "ESCALATION_REQUIRED, instead of requesting the true amount owed.",
                        override_decision="ESCALATION_REQUIRED",
                        override_refund_amount=None,
                        override_refund_id=None,
                    )
                )

    if last_refund is None and decision == "AUTO_REFUND_APPROVED":
        findings.append(
            _Finding(
                "decision is AUTO_REFUND_APPROVED but process_refund was never called.",
                override_decision="ESCALATION_REQUIRED",
                override_refund_amount=None,
                override_refund_id=None,
            )
        )

    if (
        error_calls
        and last_refund is None
        and decision not in ("CANNOT_RESOLVE", "ESCALATION_REQUIRED", "REJECTED")
    ):
        codes = ", ".join(sorted({c.result.get("error") for c in error_calls}))
        findings.append(
            _Finding(
                f"a tool returned an error ({codes}) and no refund was processed, "
                f"but decision was '{decision}'.",
                override_decision="CANNOT_RESOLVE",
                override_refund_amount=None,
                override_refund_id=None,
            )
        )

    return findings


def validate_resolution(
    resolution: Dict[str, Any], tool_calls: List[ToolCallRecord]
) -> List[str]:
    """Cross-check the stated decision against what the tools actually returned.

    Returns a list of human-readable violations (empty if consistent).
    Detection only -- see :func:`enforce_resolution` for the version that
    also corrects what it finds. Kept as its own function, with this exact
    signature and behavior, for backward compatibility and because
    detection-without-enforcement is still a legitimate, separately useful
    capability (e.g. for a caller that only wants to audit, not mutate).
    """
    return [finding.message for finding in _find_issues(resolution, tool_calls)]


def _safe_customer_response(
    decision: str, *, refund_amount: Any = None, tool_message: Optional[str] = None
) -> str:
    """A safe, generic customer-facing message for a given (corrected)
    decision. Never fabricates anything beyond what's passed in -- amounts
    and tool_message come straight from a real tool result, never invented.
    """
    if decision == "AUTO_REFUND_APPROVED":
        amount_str = f"${refund_amount:.2f}" if isinstance(refund_amount, (int, float)) else "the approved amount"
        return f"Good news -- your refund of {amount_str} has been approved and is on its way."
    if decision == "ESCALATION_REQUIRED":
        return (
            "This case needs a closer look from our operations team before we "
            "can give you a final answer -- I've escalated it now and we'll "
            "follow up shortly."
        )
    if decision == "REJECTED":
        base = "After reviewing your case, we're not able to process a refund for this order."
        if tool_message:
            base += f" {tool_message}"
        return base
    if decision == "CANNOT_RESOLVE":
        return (
            "I wasn't able to find the order or account details needed to look "
            "into this -- could you double-check the order number and get back "
            "to us?"
        )
    return "This case has been flagged for manual review."


def enforce_resolution(
    resolution: Dict[str, Any], tool_calls: List[ToolCallRecord]
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Detect AND correct any inconsistency between the stated decision and
    what the tools actually returned.

    Returns ``(corrected_resolution, validation_warnings, corrections)``.
    ``validation_warnings`` is exactly what :func:`validate_resolution` would
    return (the same findings, described). ``corrections`` describes what
    was actually changed and why -- the audit trail that keeps a correction
    visible rather than silent.

    No policy math is reimplemented here: every override either reads a
    tool's own ``status``/``approved_amount``/``refund_id`` field directly,
    or falls back to a conservative decision (``ESCALATION_REQUIRED`` /
    ``CANNOT_RESOLVE``) when the situation is ambiguous or suspicious. This
    assumes ``resolution`` has already passed :func:`validate_schema` --
    it does not itself guard against missing/malformed fields.
    """
    findings = _find_issues(resolution, tool_calls)
    warnings = [finding.message for finding in findings]

    if not findings:
        return resolution, warnings, []

    corrected = dict(resolution)
    action = dict(resolution.get("action_taken") or {})
    corrections: List[str] = []

    hard = [f for f in findings if f.override_decision is not None]

    if hard:
        winning = min(hard, key=lambda f: _DECISION_PRIORITY.get(f.override_decision, 99))
        original_decision = action.get("decision")
        refund_amount = None if winning.override_refund_amount is _UNSET else winning.override_refund_amount
        refund_id = None if winning.override_refund_id is _UNSET else winning.override_refund_id

        action["decision"] = winning.override_decision
        action["refund_amount"] = refund_amount
        action["refund_id"] = refund_id

        refund_calls = [c for c in tool_calls if c.name == "process_refund" and isinstance(c.result, dict)]
        tool_message = refund_calls[-1].result.get("message") if refund_calls else None

        corrected["customer_response"] = _safe_customer_response(
            winning.override_decision, refund_amount=refund_amount, tool_message=tool_message
        )
        corrections.append(
            f"decision overridden from {original_decision!r} to {winning.override_decision!r}: {winning.message}"
        )
    else:
        changed = False
        for finding in findings:
            if finding.override_refund_amount is not _UNSET and action.get("refund_amount") != finding.override_refund_amount:
                action["refund_amount"] = finding.override_refund_amount
                changed = True
                corrections.append(f"refund_amount corrected to {finding.override_refund_amount!r}: {finding.message}")
            if finding.override_refund_id is not _UNSET and action.get("refund_id") != finding.override_refund_id:
                action["refund_id"] = finding.override_refund_id
                changed = True
                corrections.append(f"refund_id corrected to {finding.override_refund_id!r}: {finding.message}")
        if changed:
            corrected["customer_response"] = _safe_customer_response(
                action.get("decision"), refund_amount=action.get("refund_amount")
            )

    corrected["action_taken"] = action
    return corrected, warnings, corrections
