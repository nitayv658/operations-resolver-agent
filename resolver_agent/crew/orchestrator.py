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
from typing import Any, Dict, List, Optional, Tuple

import anthropic

import multi_agent_tools as mat  # noqa: E402  (starter-kit/ is on sys.path -- see crew/__init__.py)

from ..agent import DEFAULT_MAX_RETRIES, DEFAULT_MODEL
from ..logging_utils import get_logger, log_event
from .comms.agent import CommsAgent
from .decision.agent import DecisionAgent
from .researcher.agent import ResearcherAgent
from .schemas import CrewResult, RiskReport

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


def _dispatch_fallback_security_alert(risk_report: RiskReport, case_id: str) -> Tuple[Dict[str, Any], bool, Optional[Dict[str, Any]]]:
    """Page security directly off the Researcher's own report, for a case
    where the Decision agent failed but the report already says
    ``requires_security_channel=true``.

    Without this, a Decision-stage failure is otherwise silent on exactly
    the cases that matter most: the pipeline stops, the customer gets a
    safe generic reply, and Trust & Safety never hears about a case the
    Researcher already scored as high risk -- observed for real on a live
    run (a stalled Decision call on a risk_score=90 case produced zero
    alert). This is deterministic and code-only, never a model call: it
    reuses multi_agent_tools.get_escalation_route with the exact same
    condition (risk_band == 'high' or prior_fraud_flags > 0) that
    ``requires_security_channel`` itself is defined by, so called only when
    that flag is true, it can only ever land on the fraud channel a normal
    run would also have picked.
    """
    escalation = mat.get_escalation_route(
        risk_band=risk_report.risk_band,
        requested_amount=risk_report.evidence.get("order_total_usd", 0.0),
        prior_fraud_flags=risk_report.evidence.get("prior_fraud_flags", 0),
        order_status=risk_report.evidence.get("order_status", "delivered"),
        verdict="UNKNOWN",  # no policy verdict exists -- the Decision agent never ran check_return_policy
    )
    alert_record = mat.send_slack_alert(
        channel_id=escalation["channel_id"],
        severity=escalation["severity"],
        payload={
            "order_id": risk_report.order_id,
            "user_id": risk_report.user_id,
            "risk_score": risk_report.risk_score,
            "risk_band": risk_report.risk_band,
            "triggered_rules": [f"{r['rule_id']}: {r['name']}" for r in risk_report.triggered_rules],
            "requested_amount": risk_report.evidence.get("order_total_usd", 0.0),
        },
        message=(
            f"⚠️ DECISION STAGE INCOMPLETE — order {risk_report.order_id} / "
            f"customer {risk_report.user_id}\n"
            f"Risk score: {risk_report.risk_score}/100 ({risk_report.risk_band})\n"
            "The Decision agent did not produce a verdict for this case -- routed "
            "directly from the Researcher's risk report for manual review, since "
            "it already requires the security channel."
        ),
    )
    log_event(
        _logger,
        logging.WARNING,
        "crew.decision_incomplete_fallback_alert",
        channel=escalation.get("channel"),
        delivered=bool(alert_record.get("delivered")),
        case_id=case_id,
    )
    return escalation, bool(alert_record.get("delivered")), alert_record


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
            escalation: Optional[Dict[str, Any]] = None
            alert_sent = False
            alert_record: Optional[Dict[str, Any]] = None
            fallback_note = (
                f"requires_security_channel={risk_report.requires_security_channel} -- "
                "no direct alert dispatched."
            )
            if risk_report.requires_security_channel:
                escalation, alert_sent, alert_record = _dispatch_fallback_security_alert(risk_report, case_id)
                fallback_note = (
                    f"requires_security_channel=true -- dispatched a direct security alert "
                    f"to {escalation.get('channel')} (alert_sent={alert_sent}), since the "
                    "Researcher already flagged this case for the security channel."
                )
            log_event(
                _logger,
                logging.WARNING,
                "crew.decision_incomplete_escalating",
                error=decision_result.error,
                fallback_alert_sent=alert_sent,
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
                escalation=escalation,
                alert_sent=alert_sent,
                alert_record=alert_record,
                reasoning_chain=[
                    f"Risk report: order {risk_report.order_id}, risk_band={risk_report.risk_band} "
                    f"(score {risk_report.risk_score}).",
                    f"Decision agent could not produce a decision: {decision_result.error}.",
                    fallback_note,
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
