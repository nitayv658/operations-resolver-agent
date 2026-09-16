"""Tests for resolver_agent_crewai.schemas -- the Pydantic mirror of
resolver_agent.output_tool.SUBMIT_RESOLUTION_SCHEMA used as CrewAI's
Task.output_pydantic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from resolver_agent.output_tool import DECISION_VALUES
from resolver_agent_crewai.schemas import ActionTaken, Resolution


def test_decision_literal_matches_output_tool_decision_values_exactly():
    # Guards against the two ever drifting -- see schemas.py: Decision is
    # derived from DECISION_VALUES, not a second hardcoded copy.
    allowed = ActionTaken.model_fields["decision"].annotation.__args__
    assert set(allowed) == set(DECISION_VALUES)


def test_action_taken_rejects_a_decision_outside_the_four_real_values():
    with pytest.raises(ValidationError):
        ActionTaken(tools_called=[], decision="MAYBE_REFUND_LATER")


@pytest.mark.parametrize("decision", DECISION_VALUES)
def test_action_taken_accepts_every_real_decision_value(decision):
    action = ActionTaken(tools_called=["get_order_details"], decision=decision)
    assert action.decision == decision
    assert action.refund_amount is None
    assert action.refund_id is None


def test_resolution_round_trips_through_model_dump_like_a_plain_resolution_dict():
    resolution = Resolution(
        reasoning_chain=["Customer sentiment: calm. Urgency: low.", "ORD-1001 total is $48."],
        action_taken=ActionTaken(
            tools_called=["get_order_details", "process_refund"],
            decision="AUTO_REFUND_APPROVED",
            refund_amount=48.0,
            refund_id="RF-1001-4800",
        ),
        customer_response="Good news -- your refund of $48.00 has been approved.",
    )

    dumped = resolution.model_dump()

    # This is exactly the shape resolver_agent.output_tool.validate_schema /
    # enforce_resolution expect -- a plain dict with these three top-level keys.
    assert set(dumped.keys()) == {"reasoning_chain", "action_taken", "customer_response"}
    assert dumped["action_taken"]["refund_amount"] == 48.0
