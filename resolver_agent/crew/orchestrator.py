"""OperationsCrew -- coordinates Researcher -> Decision -> Comms.

Each agent reuses resolver_agent.tool_loop.run_tool_loop unchanged, with its
own prompt and its own tool bundle (multi_agent_tools.RESEARCHER_TOOLS /
DECISION_TOOLS / COMMS_TOOLS) -- financial authority and messaging are
separated by construction, not convention: DecisionAgent's tool_registry is
the only one that can even reach process_refund, and CommsAgent's cannot
reach it at all (see starter-kit/multi_agent_tools.py's own per-role bundles
and TOOL_OWNERSHIP).

The one guardrail that lives here rather than inside a single agent: an
incomplete/failed Researcher or Decision report stops the pipeline and
escalates immediately. It does **not** re-dispatch the agent that failed --
the brief is explicit that a retry loop is the wrong response to missing
data, and each agent's own ``max_iterations`` (passed through to
run_tool_loop) already bounds a single stage's own runaway.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import anthropic

from ..agent import DEFAULT_MAX_RETRIES, DEFAULT_MODEL
from ..logging_utils import get_logger, log_event
from .comms.agent import CommsAgent
from .decision.agent import DecisionAgent
from .researcher.agent import ResearcherAgent
from .schemas import CrewResult

_logger = get_logger(__name__)


def _lookup_failure_response(error: Optional[Dict[str, Any]]) -> str:
    """A safe, generic reply when the Researcher could not resolve the order
    or user -- never fabricates a fact beyond what the failure itself says."""
    code = (error or {}).get("error")
    if code == "USER_ORDER_MISMATCH":
        return (
            "We weren't able to verify that this order belongs to the account on "
            "this ticket -- could you double-check the order number and get back "
            "to us? We've flagged this for a closer look."
        )
    return (
        "I wasn't able to find the order or account details needed to look into "
        "this -- could you double-check the order number and get back to us?"
    )


class OperationsCrew:
    """Agent 1 (Researcher & Fraud Auditor) -> Agent 2 (Decision Maker) ->
    Agent 3 (Comms & Escalation Manager)."""

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        model: str = DEFAULT_MODEL,
        max_iterations_per_agent: int = 6,
    ) -> None:
        if client is not None:
            self.client = client
        elif DEFAULT_MAX_RETRIES is not None:
            self.client = anthropic.Anthropic(max_retries=DEFAULT_MAX_RETRIES)
        else:
            self.client = anthropic.Anthropic()
        self.model = model
        self.researcher = ResearcherAgent(self.client, self.model, max_iterations_per_agent)
        self.decision_agent = DecisionAgent(self.client, self.model, max_iterations_per_agent)
        self.comms_agent = CommsAgent(self.client, self.model, max_iterations_per_agent)

    def handle_ticket(self, ticket_text: str) -> CrewResult:
        case_id = uuid.uuid4().hex[:8]
        ctx = {"case_id": case_id}

        researcher_result = self.researcher.run(ticket_text, case_id)
        if researcher_result.report is None:
            log_event(
                _logger,
                logging.WARNING,
                "crew.researcher_incomplete_escalating",
                error=researcher_result.error,
                **ctx,
            )
            return CrewResult(
                order_id=researcher_result.order_id or "UNKNOWN",
                customer_response=_lookup_failure_response(researcher_result.error),
                decision=None,
                escalation=None,
                alert_sent=False,
                alert_record=None,
                reasoning_chain=[f"Researcher could not produce a risk report: {researcher_result.error}."],
                stopped_reason="researcher_incomplete",
            )

        risk_report = researcher_result.report
        decision_result = self.decision_agent.run(risk_report, case_id)
        if decision_result.decision is None:
            log_event(
                _logger,
                logging.WARNING,
                "crew.decision_incomplete_escalating",
                error=decision_result.error,
                **ctx,
            )
            return CrewResult(
                order_id=risk_report.order_id,
                customer_response=(
                    "This case needs a closer look from our operations team before "
                    "we can give you a final answer -- we're escalating it now and "
                    "will follow up shortly."
                ),
                decision=None,
                escalation=None,
                alert_sent=False,
                alert_record=None,
                reasoning_chain=[
                    f"Risk report: order {risk_report.order_id}, risk_band={risk_report.risk_band} "
                    f"(score {risk_report.risk_score}).",
                    f"Decision agent could not produce a decision: {decision_result.error}.",
                ],
                stopped_reason="decision_incomplete",
            )

        decision = decision_result.decision
        comms_result = self.comms_agent.run(decision, case_id)

        reasoning: List[str] = [
            f"Risk report: order {risk_report.order_id}, user {risk_report.user_id}, "
            f"risk_band={risk_report.risk_band} (score {risk_report.risk_score}), "
            f"blocks_automatic_refund={risk_report.blocks_automatic_refund}.",
        ]
        reasoning.extend(researcher_result.warnings)
        reasoning.extend(f"[corrected] {c}" for c in researcher_result.corrections)
        reasoning.append(
            f"Decision: verdict={decision.verdict}, refund_status={decision.refund_status}, "
            f"requested_amount={decision.requested_amount}, approved_amount={decision.approved_amount}."
        )
        reasoning.extend(decision_result.warnings)
        reasoning.extend(f"[corrected] {c}" for c in decision_result.corrections)
        if comms_result.escalation is not None:
            reasoning.append(
                f"Escalation route: escalation_required={comms_result.escalation.get('escalation_required')}, "
                f"channel={comms_result.escalation.get('channel')}."
            )
        reasoning.append(f"Alert sent: {comms_result.alert_sent}.")

        log_event(
            _logger,
            logging.INFO,
            "crew.case_resolved",
            refund_status=decision.refund_status,
            alert_sent=comms_result.alert_sent,
            stopped_reason=comms_result.stopped_reason,
            **ctx,
        )

        return CrewResult(
            order_id=risk_report.order_id,
            customer_response=comms_result.customer_response,
            decision=decision,
            escalation=comms_result.escalation,
            alert_sent=comms_result.alert_sent,
            alert_record=comms_result.alert_record,
            reasoning_chain=reasoning,
            stopped_reason=comms_result.stopped_reason,
        )
