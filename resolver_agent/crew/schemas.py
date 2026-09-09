"""Structured handoffs passed between the three crew agents.

Part A's tools already return flat, JSON-serializable dicts specifically so
they "slot straight into a Pydantic model" (starter-kit README). These models
are that slot: each one is the *complete* contract for what one agent hands
the next, so nothing gets silently summarized or dropped crossing an agent
boundary -- the brief's own central trap (ORD-1005: policy alone says
ELIGIBLE, only the fraud report says otherwise) is exactly what's lost if a
handoff carries a verdict instead of the full upstream report.

Each stage receives the previous stage's model serialized as JSON
(``model.model_dump_json()``) in its own initial user message -- see
resolver_agent/crew/orchestrator.py -- so the model reads real structured
data, never a paraphrase of it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class RiskReport(BaseModel):
    """Agent 1 (Researcher) -> Agent 2 (Decision). Mirrors
    multi_agent_tools.audit_fraud_risk's return shape field-for-field --
    this model doesn't reinterpret the fraud engine's output, it just gives
    it a schema so it can travel."""

    order_id: str
    user_id: str
    risk_score: int
    risk_band: Literal["low", "medium", "high"]
    action_hint: str
    triggered_rules: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    blocks_automatic_refund: bool
    requires_security_channel: bool
    rulebook_version: str


class Decision(BaseModel):
    """Agent 2 (Decision Maker) -> Agent 3 (Comms). Carries the
    ``RiskReport`` through IN FULL (not just its band) -- Agent 3 needs the
    real risk_score/triggered_rules to route and to build an alert payload,
    and losing it here would silently defeat the point of Agent 1's work."""

    order_id: str
    user_id: str
    verdict: str  # check_return_policy's verdict, e.g. "ELIGIBLE"
    eligible: bool
    refund_status: Literal["APPROVED", "REJECTED", "ESCALATION_REQUIRED"]
    requested_amount: float
    approved_amount: Optional[float]
    refund_id: Optional[str]
    applicable_policies: List[str]
    rationale: str
    risk_report: RiskReport


class CrewResult(BaseModel):
    """What ``OperationsCrew.handle_ticket`` returns to the caller. Not
    itself a handoff (nothing consumes this downstream) -- the final,
    caller-facing shape, deliberately close in spirit to Part A's
    ``ResolverAgent.resolve()`` return dict (decision + customer_response +
    an auditable trace) but wrapping the crew's own richer ``Decision``.

    ``decision`` is ``None`` on the crew's own stop-condition paths: the
    Researcher's report was missing/incomplete, or the Decision agent never
    produced a valid decision. Neither is a retry loop -- see
    ``orchestrator.OperationsCrew.handle_ticket``."""

    order_id: str
    customer_response: str
    decision: Optional[Decision] = None
    escalation: Optional[Dict[str, Any]] = None  # get_escalation_route's result
    alert_sent: bool = False
    alert_record: Optional[Dict[str, Any]] = None  # send_slack_alert's result
    reasoning_chain: List[str]  # concatenated trace across all 3 agents
    stopped_reason: str  # "stop" | "max_iterations" | "researcher_incomplete" | "decision_incomplete" | "api_error"
