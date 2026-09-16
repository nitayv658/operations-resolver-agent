"""The structured-output contract for CrewResolverAgent, as a Pydantic model.

Mirrors resolver_agent.output_tool.SUBMIT_RESOLUTION_SCHEMA field-for-field.
Part 1 exposes this shape *as a tool* (submit_resolution) because the
Anthropic tool-calling loop has no other way to force structured output.
CrewAI's Task takes a `output_pydantic` model instead and parses the agent's
final answer into it directly -- same contract, different mechanism (see the
root README's CrewAI-port section for what that mechanism swap actually
changes).

Deliberately reuses resolver_agent.output_tool.validate_schema and
enforce_resolution against the dict form of this model (via
Resolution.model_dump()) rather than re-implementing that cross-check logic
here -- it is pure, framework-agnostic dict logic.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from resolver_agent.output_tool import DECISION_VALUES

# Derived from resolver_agent.output_tool.DECISION_VALUES, not hardcoded a
# second time -- a new decision value there must not be able to drift out of
# sync with this Pydantic schema without this Literal changing too.
Decision = Literal[tuple(DECISION_VALUES)]  # type: ignore[valid-type]


class ActionTaken(BaseModel):
    tools_called: List[str] = Field(
        default_factory=list,
        description="Names of every GlobalCart tool called, in order.",
    )
    decision: Decision = Field(
        description=(
            "AUTO_REFUND_APPROVED: process_refund returned APPROVED. REJECTED: the "
            "claim is not eligible (policy says no). ESCALATION_REQUIRED: "
            "process_refund returned ESCALATION_REQUIRED, or the case needs a human "
            "for any other reason. CANNOT_RESOLVE: the order/user needed to decide "
            "could not be found."
        )
    )
    refund_amount: Optional[float] = Field(
        default=None, description="process_refund's approved_amount, or null if none was approved."
    )
    refund_id: Optional[str] = Field(
        default=None, description="process_refund's refund_id, or null if none was approved."
    )


class Resolution(BaseModel):
    reasoning_chain: List[str] = Field(
        description=(
            "Ordered list of concrete facts the decision is based on -- quote real "
            "values and policy ids actually seen in tool results (order id, amounts, "
            "dates, tier, policy ids). Do not write generic statements that could "
            "apply to any ticket."
        )
    )
    action_taken: ActionTaken
    customer_response: str = Field(
        description=(
            "The reply to send the customer. Must match action_taken.decision exactly "
            "-- never describe a refund as done unless decision is "
            "AUTO_REFUND_APPROVED. Match the customer's tone and language."
        )
    )
