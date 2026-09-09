"""The Comms agent's forced-final handoff: submit_comms_result.

Deliberately the smallest of the three handoff schemas -- only one field.
Everything else about this stage's outcome (whether an alert was sent, to
which channel, with what payload) is *derived from the real tool_calls* the
agent actually made (see agent.py), never trusted from the model's own
self-report -- the same "the model's job is language, the system's job is
truth" split resolver_agent/output_tool.py already applies to
customer_response vs decision.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..schemas import Decision

SUBMIT_COMMS_RESULT_TOOL_NAME = "submit_comms_result"

SUBMIT_COMMS_RESULT_SCHEMA: Dict[str, Any] = {
    "name": SUBMIT_COMMS_RESULT_TOOL_NAME,
    "description": (
        "Record the customer-facing reply for this case and end your turn. "
        "Call this exactly once, as your last action, after you have "
        "checked the escalation route and sent an alert if one was "
        "required. Do not call any other tool after this one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_response": {
                "type": "string",
                "description": (
                    "The reply to send the customer. Must match the decision's "
                    "refund_status exactly, and must never mention fraud, risk "
                    "scores, or a security review. Match the customer's tone "
                    "and language."
                ),
            },
        },
        "required": ["customer_response"],
    },
}


def validate_schema(result: Dict[str, Any]) -> List[str]:
    """Structural/type completeness check, independent of the API's own
    tool-schema enforcement -- see the module docstring."""
    errors: List[str] = []
    response = result.get("customer_response")
    if not isinstance(response, str) or not response.strip():
        errors.append("customer_response is missing, not a string, or empty.")
    return errors


_CURRENCY_RE = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_REFUND_ID_RE = re.compile(r"\bRF-[A-Za-z0-9-]+\b")


def find_stale_refund_detail(customer_response: str, decision: Decision) -> Optional[str]:
    """Cross-check ``customer_response`` against the decision's own vetted
    fields -- ``requested_amount``/``approved_amount``/``refund_id`` -- the
    same "the model's job is language, the system's job is truth" split this
    module's docstring describes, extended to cover a real leak found live:
    ``decision.rationale`` is the Decision agent's own free-text working
    notes and can describe an earlier, since-corrected process_refund
    attempt (e.g. one capped at the auto-refund limit before the case was
    escalated) -- the code guardrail in ``decision/output_tool.py`` nulls
    out ``approved_amount``/``refund_id`` when it overrides ``refund_status``,
    but it never touches ``rationale``, so that stale figure survives into
    Comms's input untouched. A prompt instruction telling Comms not to
    trust it is not a guarantee (observed live: a smaller model repeated the
    exact stale amount and refund_id anyway) -- this is the deterministic
    backstop.

    Returns a human-readable violation message, or ``None`` if the response
    only cites amounts/refund_id the decision actually vetted.
    """
    known_amounts = {round(decision.requested_amount, 2)}
    if decision.approved_amount is not None:
        known_amounts.add(round(decision.approved_amount, 2))

    for match in _CURRENCY_RE.finditer(customer_response):
        amount = float(match.group(1).replace(",", ""))
        if round(amount, 2) not in known_amounts:
            return (
                f"customer_response cites ${match.group(1)}, which matches neither "
                f"decision.requested_amount ({decision.requested_amount}) nor "
                f"decision.approved_amount ({decision.approved_amount}) -- likely a stale "
                "figure repeated from decision.rationale."
            )

    for match in _REFUND_ID_RE.finditer(customer_response):
        if match.group(0) != decision.refund_id:
            return (
                f"customer_response cites refund_id {match.group(0)!r}, which does not match "
                f"decision.refund_id ({decision.refund_id!r}) -- likely a stale refund_id "
                "repeated from decision.rationale."
            )

    return None
