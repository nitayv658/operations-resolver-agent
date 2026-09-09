"""The Decision agent's forced-final handoff: submit_decision.

Same three-layer recipe as resolver_agent/output_tool.py:

1. :func:`validate_schema` -- structural/type completeness, no tool_calls
   needed.
2. :func:`_find_issues` -- cross-checks the stated refund_status against
   what the tools (check_return_policy, process_refund) actually returned,
   PLUS the one check unique to this crew: the incoming RiskReport's
   ``blocks_automatic_refund`` is a hard block that a clean policy verdict
   cannot override. This is the literal ORD-1005 guardrail (policy says
   ELIGIBLE, the fraud engine says high) -- if it only lived in the prompt,
   a model could talk itself past it.
3. :func:`enforce_decision` -- uses the same findings as (2) but *acts* on
   them, deterministically, before the decision ever reaches Agent 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...tool_loop import ToolCallRecord
from ..schemas import RiskReport

SUBMIT_DECISION_TOOL_NAME = "submit_decision"

REFUND_STATUS_VALUES = ["APPROVED", "REJECTED", "ESCALATION_REQUIRED"]

# Most-conservative-wins tie-break, same principle as
# resolver_agent.output_tool._DECISION_PRIORITY.
_STATUS_PRIORITY = {"ESCALATION_REQUIRED": 0, "REJECTED": 1, "APPROVED": 2}

_UNSET = object()

SUBMIT_DECISION_SCHEMA: Dict[str, Any] = {
    "name": SUBMIT_DECISION_TOOL_NAME,
    "description": (
        "Record the final operational decision for this case and end your "
        "turn. Call this exactly once, as your last action, only after you "
        "have consulted policy and (if appropriate) attempted the refund. "
        "Do not call any other tool after this one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "user_id": {"type": "string"},
            "verdict": {"type": "string", "description": "check_return_policy's verdict, e.g. 'ELIGIBLE'."},
            "eligible": {"type": "boolean", "description": "check_return_policy's eligible field."},
            "refund_status": {"type": "string", "enum": REFUND_STATUS_VALUES},
            "requested_amount": {"type": "number", "description": "The amount actually requested from process_refund."},
            "approved_amount": {"type": ["number", "null"], "description": "process_refund's approved_amount, or null."},
            "refund_id": {"type": ["string", "null"], "description": "process_refund's refund_id, or null."},
            "applicable_policies": {"type": "array", "items": {"type": "string"}},
            "rationale": {
                "type": "string",
                "description": "Concrete facts this decision is based on -- real policy ids and tool results, not generic phrasing.",
            },
        },
        "required": [
            "order_id",
            "user_id",
            "verdict",
            "eligible",
            "refund_status",
            "requested_amount",
            "applicable_policies",
            "rationale",
        ],
    },
}


def validate_schema(decision: Dict[str, Any]) -> List[str]:
    """Structural/type completeness check, independent of the API's own
    tool-schema enforcement -- see the module docstring."""
    errors: List[str] = []

    if not isinstance(decision.get("order_id"), str) or not decision["order_id"].strip():
        errors.append("order_id is missing or not a non-empty string.")
    if not isinstance(decision.get("user_id"), str) or not decision["user_id"].strip():
        errors.append("user_id is missing or not a non-empty string.")
    if not isinstance(decision.get("verdict"), str) or not decision["verdict"].strip():
        errors.append("verdict is missing or not a non-empty string.")
    if not isinstance(decision.get("eligible"), bool):
        errors.append("eligible is missing or is not a boolean.")
    if decision.get("refund_status") not in REFUND_STATUS_VALUES:
        errors.append(
            f"refund_status is missing or not one of {REFUND_STATUS_VALUES}: got {decision.get('refund_status')!r}."
        )
    if not isinstance(decision.get("requested_amount"), (int, float)) or isinstance(decision.get("requested_amount"), bool):
        errors.append("requested_amount is missing or is not a number.")
    if not isinstance(decision.get("applicable_policies"), list):
        errors.append("applicable_policies is missing or is not a list.")
    if not isinstance(decision.get("rationale"), str) or not decision["rationale"].strip():
        errors.append("rationale is missing, not a string, or empty.")

    return errors


@dataclass
class _Finding:
    message: str
    override_status: Optional[str] = None
    override_approved_amount: Any = _UNSET
    override_refund_id: Any = _UNSET
    override_requested_amount: Any = _UNSET


def _find_issues(
    decision: Dict[str, Any], tool_calls: List[ToolCallRecord], risk_report: RiskReport
) -> List[_Finding]:
    findings: List[_Finding] = []
    status = decision.get("refund_status")

    refund_calls = [c for c in tool_calls if c.name == "process_refund" and isinstance(c.result, dict)]
    last_refund = refund_calls[-1].result if refund_calls else None

    # --- the ORD-1005 guardrail: a fraud block cannot be approved past ---- #
    if risk_report.blocks_automatic_refund and status == "APPROVED":
        findings.append(
            _Finding(
                f"risk_report.blocks_automatic_refund is True (risk_band={risk_report.risk_band!r}, "
                f"risk_score={risk_report.risk_score}) but refund_status was 'APPROVED' -- a fraud "
                "block overrides a clean policy verdict.",
                override_status="ESCALATION_REQUIRED",
                override_approved_amount=None,
                override_refund_id=None,
            )
        )

    # --- cross-check against what process_refund actually returned ------- #
    if last_refund is not None:
        tool_status = last_refund.get("status")
        if tool_status != status and tool_status in REFUND_STATUS_VALUES:
            mapped = tool_status if tool_status != "APPROVED" or not risk_report.blocks_automatic_refund else "ESCALATION_REQUIRED"
            findings.append(
                _Finding(
                    f"process_refund returned status '{tool_status}' but refund_status was '{status}'.",
                    override_status=mapped,
                    override_approved_amount=(last_refund.get("approved_amount") if mapped == "APPROVED" else None),
                    override_refund_id=(last_refund.get("refund_id") if mapped == "APPROVED" else None),
                )
            )
        elif tool_status == "APPROVED":
            if decision.get("approved_amount") != last_refund.get("approved_amount"):
                findings.append(
                    _Finding(
                        f"approved_amount ({decision.get('approved_amount')}) does not match "
                        f"process_refund's approved_amount ({last_refund.get('approved_amount')}).",
                        override_approved_amount=last_refund.get("approved_amount"),
                    )
                )
            if decision.get("refund_id") != last_refund.get("refund_id"):
                findings.append(
                    _Finding(
                        f"refund_id ({decision.get('refund_id')}) does not match process_refund's "
                        f"refund_id ({last_refund.get('refund_id')}).",
                        override_refund_id=last_refund.get("refund_id"),
                    )
                )

            # The same trap resolver_agent.output_tool guards against: an
            # agent that requests exactly the cap instead of the real amount
            # owed gets a clean APPROVED back (process_refund enforces its
            # cap, not intent), which can't be told apart from an honest
            # claim that happens to equal the cap without an independent
            # ground truth for the real amount owed. The Decision agent has
            # no get_order_details tool of its own, but risk_report.evidence
            # (passed through from the Researcher in full) carries
            # order_total_usd -- exactly the ground truth this check needs.
            requested = last_refund.get("requested_amount")
            cap = last_refund.get("auto_refund_cap_usd")
            order_total = risk_report.evidence.get("order_total_usd")
            if (
                requested is not None
                and cap is not None
                and order_total is not None
                and requested < order_total
                and requested == cap
            ):
                findings.append(
                    _Finding(
                        f"process_refund was called with amount={requested} == the auto-refund cap "
                        f"({cap}), below the real order total from the risk report "
                        f"({order_total}) -- looks like an under-request to dodge escalation, "
                        "instead of requesting the true amount owed.",
                        override_status="ESCALATION_REQUIRED",
                        override_approved_amount=None,
                        override_refund_id=None,
                        # Also correct requested_amount to the real amount owed --
                        # Comms routes off this field (get_escalation_route's
                        # requested_amount param), and leaving it at the
                        # under-requested figure would make a $150 claim look
                        # like a harmless $50 one to Agent 3, silently
                        # defeating the very override this finding just made.
                        override_requested_amount=order_total,
                    )
                )
    elif status == "APPROVED":
        findings.append(
            _Finding(
                "refund_status is APPROVED but process_refund was never called.",
                override_status="ESCALATION_REQUIRED",
                override_approved_amount=None,
                override_refund_id=None,
            )
        )

    return findings


def enforce_decision(
    decision: Dict[str, Any], tool_calls: List[ToolCallRecord], risk_report: RiskReport
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Detect AND correct any inconsistency between the stated refund_status
    and (a) what process_refund actually returned, (b) the incoming risk
    report's blocks_automatic_refund flag. Returns
    ``(corrected_decision, validation_warnings, corrections)``.

    Assumes ``decision`` already passed :func:`validate_schema`.
    """
    findings = _find_issues(decision, tool_calls, risk_report)
    warnings = [f.message for f in findings]

    if not findings:
        return decision, warnings, []

    corrected = dict(decision)
    corrections: List[str] = []

    hard = [f for f in findings if f.override_status is not None]
    if hard:
        winning = min(hard, key=lambda f: _STATUS_PRIORITY.get(f.override_status, 99))
        original = decision.get("refund_status")
        approved_amount = None if winning.override_approved_amount is _UNSET else winning.override_approved_amount
        refund_id = None if winning.override_refund_id is _UNSET else winning.override_refund_id
        corrected["refund_status"] = winning.override_status
        corrected["approved_amount"] = approved_amount
        corrected["refund_id"] = refund_id
        corrections.append(f"refund_status overridden from {original!r} to {winning.override_status!r}: {winning.message}")
        if winning.override_requested_amount is not _UNSET:
            original_requested = decision.get("requested_amount")
            corrected["requested_amount"] = winning.override_requested_amount
            corrections.append(
                f"requested_amount corrected from {original_requested!r} to "
                f"{winning.override_requested_amount!r} so downstream routing sees the real amount owed: {winning.message}"
            )
    else:
        for f in findings:
            if f.override_approved_amount is not _UNSET and corrected.get("approved_amount") != f.override_approved_amount:
                corrected["approved_amount"] = f.override_approved_amount
                corrections.append(f"approved_amount corrected to {f.override_approved_amount!r}: {f.message}")
            if f.override_refund_id is not _UNSET and corrected.get("refund_id") != f.override_refund_id:
                corrected["refund_id"] = f.override_refund_id
                corrections.append(f"refund_id corrected to {f.override_refund_id!r}: {f.message}")

    return corrected, warnings, corrections
