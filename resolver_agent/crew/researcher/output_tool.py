"""The Researcher's forced-final handoff: submit_risk_report.

Same recipe as resolver_agent/output_tool.py's submit_resolution: the
required output is exposed as a tool so its arguments arrive already
schema-validated by the API, and that constraint is independently
re-checked here rather than trusted alone -- an out-of-enum status or a
missing risk field would still reach us if the model ever produced one.

Two shapes are legitimate, distinguished by ``status``:

- ``"OK"`` -- the order and user were found and audit_fraud_risk produced a
  real report; every risk field is required.
- ``"LOOKUP_FAILED"`` -- get_order_details / get_user_profile /
  audit_fraud_risk returned an error (ORDER_NOT_FOUND, USER_NOT_FOUND,
  USER_ORDER_MISMATCH); the risk fields are meaningless and must not be
  guessed at, so only ``error`` is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ...tool_loop import ToolCallRecord

SUBMIT_RISK_REPORT_TOOL_NAME = "submit_risk_report"

STATUS_VALUES = ["OK", "LOOKUP_FAILED"]

SUBMIT_RISK_REPORT_SCHEMA: Dict[str, Any] = {
    "name": SUBMIT_RISK_REPORT_TOOL_NAME,
    "description": (
        "Record this case's risk report and end your investigation. Call "
        "this exactly once, as your last action. Use status='OK' with every "
        "risk field filled in from a real audit_fraud_risk result, or "
        "status='LOOKUP_FAILED' with the error you actually got if the "
        "order or user could not be resolved. Do not call any other tool "
        "after this one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": STATUS_VALUES},
            "order_id": {"type": "string"},
            "user_id": {"type": "string"},
            "risk_score": {"type": "integer", "description": "audit_fraud_risk's risk_score, 0-100."},
            "risk_band": {"type": "string", "enum": ["low", "medium", "high"]},
            "action_hint": {"type": "string"},
            "triggered_rules": {
                "type": "array",
                "items": {"type": "object"},
                "description": "audit_fraud_risk's triggered_rules, verbatim.",
            },
            "evidence": {
                "type": "object",
                "description": "audit_fraud_risk's evidence dict, verbatim.",
            },
            "blocks_automatic_refund": {"type": "boolean"},
            "requires_security_channel": {"type": "boolean"},
            "rulebook_version": {"type": "string"},
            "error": {
                "type": "object",
                "description": "The failing tool's own {error, message} dict. Required when status='LOOKUP_FAILED'.",
            },
        },
        "required": ["status", "order_id"],
    },
}

_OK_REQUIRED_FIELDS = [
    "user_id",
    "risk_score",
    "risk_band",
    "action_hint",
    "triggered_rules",
    "evidence",
    "blocks_automatic_refund",
    "requires_security_channel",
    "rulebook_version",
]


def validate_schema(report: Dict[str, Any]) -> List[str]:
    """Structural/type completeness check, independent of the Anthropic
    API's own tool-schema enforcement -- see the module docstring for why
    this isn't just trusted.

    Returns a list of human-readable errors (empty if the shape is valid).
    """
    errors: List[str] = []

    status = report.get("status")
    if status not in STATUS_VALUES:
        errors.append(f"status is missing or not one of {STATUS_VALUES}: got {status!r}.")
        return errors  # nothing else can be meaningfully checked without a valid status

    if not isinstance(report.get("order_id"), str) or not report["order_id"].strip():
        errors.append("order_id is missing or not a non-empty string.")

    if status == "LOOKUP_FAILED":
        error = report.get("error")
        if not isinstance(error, dict) or "error" not in error:
            errors.append("status is LOOKUP_FAILED but error is missing or not an {error, message} dict.")
        return errors

    # status == "OK"
    if not isinstance(report.get("user_id"), str) or not report["user_id"].strip():
        errors.append("user_id is missing or not a non-empty string.")
    if not isinstance(report.get("risk_score"), int) or isinstance(report.get("risk_score"), bool):
        errors.append("risk_score is missing or not an integer.")
    if report.get("risk_band") not in ("low", "medium", "high"):
        errors.append(f"risk_band is missing or not one of low/medium/high: got {report.get('risk_band')!r}.")
    if not isinstance(report.get("triggered_rules"), list):
        errors.append("triggered_rules is missing or is not a list.")
    if not isinstance(report.get("evidence"), dict):
        errors.append("evidence is missing or is not an object.")
    if not isinstance(report.get("blocks_automatic_refund"), bool):
        errors.append("blocks_automatic_refund is missing or is not a boolean.")
    if not isinstance(report.get("requires_security_channel"), bool):
        errors.append("requires_security_channel is missing or is not a boolean.")
    if not isinstance(report.get("rulebook_version"), str):
        errors.append("rulebook_version is missing or is not a string.")

    return errors


# Fields the model must relay verbatim from audit_fraud_risk -- see
# enforce_risk_report. Deliberately excludes order_id/user_id, which are
# cross-checked separately since audit_fraud_risk's own resolved
# ("order_id", "user_id") pair is the ground truth for those too.
_GROUND_TRUTH_FIELDS = (
    "risk_score",
    "risk_band",
    "action_hint",
    "triggered_rules",
    "evidence",
    "blocks_automatic_refund",
    "requires_security_channel",
    "rulebook_version",
)


@dataclass
class _Finding:
    message: str
    field: str
    value: Any


def enforce_risk_report(
    report: Dict[str, Any], tool_calls: List[ToolCallRecord]
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Cross-check a status='OK' report against the real audit_fraud_risk
    result and correct any drift before it travels downstream.

    audit_fraud_risk is a deterministic rule engine (RESEARCHER_PROMPT rule
    2: "never adjust, round, soften, or overrule the band"), but that is a
    prompt instruction -- nothing stops a model from mis-transcribing a
    field on the way to submit_risk_report. This is the code-level guarantee
    that closes that gap, the same way
    resolver_agent.output_tool.enforce_resolution cross-checks a stated
    decision against process_refund's real result, and
    crew.decision.output_tool.enforce_decision cross-checks refund_status
    against process_refund and the incoming risk report.

    Assumes ``report`` already passed :func:`validate_schema` and has
    ``status == "OK"``. Returns ``(corrected_report, warnings, corrections)``.
    """
    audit_calls = [c for c in tool_calls if c.name == "audit_fraud_risk" and isinstance(c.result, dict)]
    successful = [c.result for c in audit_calls if "error" not in c.result]
    last = successful[-1] if successful else None

    if last is None:
        # status is OK but there is no real audit_fraud_risk result to back
        # it -- exactly the "claimed a value with nothing to support it"
        # shape resolver_agent.output_tool._find_issues treats as
        # conservative-escalate rather than a guess at the real answer.
        corrected = dict(report)
        corrected["status"] = "LOOKUP_FAILED"
        corrected["error"] = {
            "error": "NO_FRAUD_AUDIT",
            "message": "status was 'OK' but audit_fraud_risk was never successfully called.",
        }
        message = "status was 'OK' but audit_fraud_risk was never called -- downgraded to LOOKUP_FAILED."
        return corrected, [message], [message]

    corrected = dict(report)
    warnings: List[str] = []
    corrections: List[str] = []

    for field_name in ("order_id", "user_id", *_GROUND_TRUTH_FIELDS):
        truth = last.get(field_name)
        if corrected.get(field_name) != truth:
            warnings.append(f"{field_name} ({corrected.get(field_name)!r}) did not match audit_fraud_risk's own result ({truth!r}).")
            corrections.append(f"{field_name} corrected to {truth!r} to match audit_fraud_risk's own result.")
            corrected[field_name] = truth

    return corrected, warnings, corrections
