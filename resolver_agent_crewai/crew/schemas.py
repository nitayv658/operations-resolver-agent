"""Pydantic mirrors of Part 2's submit_risk_report / submit_decision tool
schemas, used as CrewAI's output_pydantic for the Researcher/Decision stages.

Not resolver_agent.crew.schemas.RiskReport / Decision directly -- two real
shape mismatches make that unsafe:

  * The Researcher's forced output is a status-discriminated union
    ("OK" with every risk field filled in, or "LOOKUP_FAILED" with just an
    error) -- RiskReport has no status/error fields at all, so it cannot
    express a lookup failure.
  * The Decision's forced output excludes risk_report entirely -- that gets
    attached by *code* afterward from the already-known upstream RiskReport
    (see crew/orchestrator.py), not re-emitted by the model. Asking the LLM
    to reproduce the entire upstream report verbatim as part of its own
    output would be wasteful and a real transcription-drift risk.

resolver_agent.crew.{researcher,decision}.output_tool.validate_schema /
enforce_risk_report / enforce_decision are reused completely unmodified
against these models' .model_dump() dicts -- same "the schema forced on the
model differs from the clean object passed downstream" shape Part 1's own
schemas.py exists for, just for a different reason this time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ResearcherOutput(BaseModel):
    status: Literal["OK", "LOOKUP_FAILED"] = Field(
        description=(
            "'OK' if the order and user were found and audit_fraud_risk produced a "
            "real report -- fill in every risk field from it verbatim. "
            "'LOOKUP_FAILED' if the order/user could not be resolved, or "
            "audit_fraud_risk returned USER_ORDER_MISMATCH -- fill in only error, "
            "never guess at risk fields you never actually got."
        )
    )
    order_id: str
    user_id: Optional[str] = None
    risk_score: Optional[int] = Field(default=None, description="audit_fraud_risk's risk_score, 0-100.")
    risk_band: Optional[Literal["low", "medium", "high"]] = None
    action_hint: Optional[str] = None
    triggered_rules: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="audit_fraud_risk's triggered_rules, verbatim."
    )
    evidence: Optional[Dict[str, Any]] = Field(default=None, description="audit_fraud_risk's evidence dict, verbatim.")
    blocks_automatic_refund: Optional[bool] = None
    requires_security_channel: Optional[bool] = None
    rulebook_version: Optional[str] = None
    error: Optional[Dict[str, Any]] = Field(
        default=None, description="The failing tool's own {error, message} dict. Required when status='LOOKUP_FAILED'."
    )


class CommsOutput(BaseModel):
    """The Comms stage's forced output -- deliberately still a Pydantic
    model despite being a single field: without output_pydantic forcing the
    model to isolate exactly this field, its raw final-answer text
    sometimes leaked its own preamble/scratchpad ahead of the actual reply
    (observed live, e.g. "I'll now write the customer reply: ---\n\n...").
    """

    customer_response: str = Field(
        description="The reply to send the customer -- just the reply text, no other commentary."
    )


class DecisionOutput(BaseModel):
    order_id: str
    user_id: str
    verdict: str = Field(description="check_return_policy's verdict, e.g. 'ELIGIBLE'.")
    eligible: bool = Field(description="check_return_policy's eligible field.")
    refund_status: Literal["APPROVED", "REJECTED", "ESCALATION_REQUIRED"]
    requested_amount: float = Field(description="The amount actually requested from process_refund.")
    approved_amount: Optional[float] = Field(default=None, description="process_refund's approved_amount, or null.")
    refund_id: Optional[str] = Field(default=None, description="process_refund's refund_id, or null.")
    applicable_policies: List[str]
    rationale: str = Field(
        description="Concrete facts this decision is based on -- real policy ids and tool results, not generic phrasing."
    )
