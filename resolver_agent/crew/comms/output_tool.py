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

from typing import Any, Dict, List

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
