"""CrewAIOperationsCrew -- Part 2's Researcher/Decision/Comms pipeline,
rebuilt on CrewAI instead of the hand-rolled tool_loop, kept side by side
with resolver_agent.crew.orchestrator.OperationsCrew for comparison.

Same public contract: .handle_ticket(ticket_text) -> CrewResult (the exact
same resolver_agent.crew.schemas.CrewResult), same control flow -- a missing
Researcher/Decision result stops the pipeline and escalates immediately,
never re-dispatches (see OperationsCrew.handle_ticket's own docstring for
why); a Decision-stage failure on a report that already
requires_security_channel pages security directly via the same deterministic,
code-only _dispatch_fallback_security_alert.

Reuses everything that isn't loop-specific, unmodified: the researcher/
decision output_tool.py's validate_schema/enforce_*, comms/output_tool.py's
find_stale_refund_detail/find_premature_approval_language,
comms/agent.py's _guarded_registry/_safe_customer_response/
CommsAgent._last_successful, and orchestrator.py's own
_lookup_failure_response/_dispatch_fallback_security_alert/
DEFAULT_COMMS_MODEL. Even the per-agent Result dataclasses (ResearcherResult,
DecisionResult, CommsResult) are reused directly -- they're plain data, not
coupled to how they were produced.

What's genuinely new here (see resolver_agent_crewai/crew/schemas.py for
why): ResearcherOutput/DecisionOutput are local Pydantic mirrors of
submit_risk_report/submit_decision's tool schemas, used as CrewAI's
output_pydantic -- not resolver_agent.crew.schemas.RiskReport/Decision
directly, which are the wrong shape for what the model is actually forced
to produce at each stage.

Mechanism-level differences from the hand-rolled version, same categories
as resolver_agent_crewai/agent.py's (see the root README's CrewAI-port
section): no forced-tool-call safety net (CrewAI's Task.output_pydantic
instead), no automatic tool-call trace (each build_*_tools() wrapper builds
its own call_log), anthropic.APIError caught per-stage instead of
tool_loop.ModelAPIError, no prompt-caching hook wired up, and
stopped_reason collapsing "stop" vs a hit iteration budget into one value.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

import anthropic  # noqa: E402
import pydantic  # noqa: E402
from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402

from resolver_agent.agent import DEFAULT_MODEL  # noqa: E402
from resolver_agent.crew.comms.agent import (  # noqa: E402
    CommsAgent,
    CommsResult,
    _safe_customer_response,
)
from resolver_agent.crew.comms.output_tool import (  # noqa: E402
    find_premature_approval_language,
    find_stale_refund_detail,
)
from resolver_agent.crew.decision.agent import DecisionResult  # noqa: E402
from resolver_agent.crew.decision.output_tool import enforce_decision  # noqa: E402
from resolver_agent.crew.decision.output_tool import validate_schema as _validate_decision_schema  # noqa: E402
from resolver_agent.crew.orchestrator import (  # noqa: E402
    DEFAULT_COMMS_MODEL,
    _dispatch_fallback_security_alert,
    _lookup_failure_response,
)
from resolver_agent.crew.researcher.agent import ResearcherResult  # noqa: E402
from resolver_agent.crew.researcher.output_tool import enforce_risk_report  # noqa: E402
from resolver_agent.crew.researcher.output_tool import validate_schema as _validate_researcher_schema  # noqa: E402
from resolver_agent.crew.schemas import CrewResult, Decision, RiskReport  # noqa: E402
from resolver_agent.logging_utils import get_logger, log_event  # noqa: E402
from resolver_agent.tool_loop import ToolCallRecord  # noqa: E402

from ..agent import _litellm_model  # noqa: E402  (Part 1's port -- same anthropic/-prefix helper)
from .prompts import (  # noqa: E402
    COMMS_BACKSTORY,
    COMMS_GOAL,
    COMMS_ROLE,
    COMMS_TASK_DESCRIPTION,
    COMMS_TASK_EXPECTED_OUTPUT,
    DECISION_BACKSTORY,
    DECISION_GOAL,
    DECISION_ROLE,
    DECISION_TASK_DESCRIPTION,
    DECISION_TASK_EXPECTED_OUTPUT,
    RESEARCHER_BACKSTORY,
    RESEARCHER_GOAL,
    RESEARCHER_ROLE,
    RESEARCHER_TASK_DESCRIPTION,
    RESEARCHER_TASK_EXPECTED_OUTPUT,
)
from .schemas import CommsOutput, DecisionOutput, ResearcherOutput
from .tools import build_comms_tools, build_decision_tools, build_researcher_tools

# Nested under "resolver_agent" -- see resolver_agent_crewai/agent.py's own
# comment on this: configure_logging() only attaches its stderr handler to
# exactly the "resolver_agent" logger tree.
_logger = get_logger("resolver_agent.crewai.crew")

_MAX_TOKENS = 4096  # mirrors tool_loop.py's own default -- see agent.py's comment on this

# Type-safe placeholders for ResearcherOutput's "OK"-required fields -- see
# _backfill_researcher_ok_fields.
_OK_FIELD_PLACEHOLDERS: Dict[str, Any] = {
    "user_id": "UNKNOWN",
    "risk_score": 0,
    "risk_band": "low",
    "action_hint": "",
    "triggered_rules": [],
    "evidence": {},
    "blocks_automatic_refund": False,
    "requires_security_channel": False,
    "rulebook_version": "",
}


def _backfill_researcher_ok_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in a type-safe placeholder for any "OK"-required field the
    model omitted, observed live: CrewAI's structured-output parsing
    sometimes lets the model skip a field it should have transcribed
    verbatim from audit_fraud_risk (dropping `evidence`/`triggered_rules`,
    or worse). ResearcherOutput has to allow that at the type level --
    these fields are only *actually* required when status=="OK", which a
    single Pydantic model can't express as a hard requirement without
    duplicating the whole LOOKUP_FAILED branch as a separate model (see
    schemas.py).

    Rather than let that benign omission fail validate_schema's purely
    *structural* check outright (a different failure mode than "present
    but wrong", which enforce_risk_report already handles), backfill it so
    it passes structurally. Safe to do unconditionally: enforce_risk_report
    unconditionally overwrites every one of these fields with the real
    audit_fraud_risk result whenever one was actually, successfully called
    (and downgrades to LOOKUP_FAILED otherwise) -- so the exact placeholder
    value never matters, only its type.
    """
    if raw.get("status") != "OK":
        return raw

    filled = dict(raw)
    for field_name, placeholder in _OK_FIELD_PLACEHOLDERS.items():
        value = filled.get(field_name)
        if field_name in ("user_id", "rulebook_version") and (not isinstance(value, str) or not value.strip()):
            filled[field_name] = placeholder
        elif field_name == "action_hint" and not isinstance(value, str):
            filled[field_name] = placeholder
        elif field_name == "risk_score" and (not isinstance(value, int) or isinstance(value, bool)):
            filled[field_name] = placeholder
        elif field_name == "risk_band" and value not in ("low", "medium", "high"):
            filled[field_name] = placeholder
        elif field_name == "triggered_rules" and not isinstance(value, list):
            filled[field_name] = placeholder
        elif field_name == "evidence" and not isinstance(value, dict):
            filled[field_name] = placeholder
        elif field_name in ("blocks_automatic_refund", "requires_security_channel") and not isinstance(value, bool):
            filled[field_name] = placeholder
    return filled


class CrewAIOperationsCrew:
    """Agent 1 (Researcher & Fraud Auditor) -> Agent 2 (Decision Maker) ->
    Agent 3 (Comms & Escalation Manager) -- the CrewAI-framework counterpart
    to resolver_agent.crew.orchestrator.OperationsCrew.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        researcher_model: Optional[str] = None,
        decision_model: Optional[str] = None,
        comms_model: Optional[str] = None,
        max_iterations_per_agent: int = 6,
    ) -> None:
        # Same resolution order as OperationsCrew.__init__: explicit kwarg,
        # then the agent's own env var, then the shared `model` -- except
        # Comms, which is never chained to `model` (see DEFAULT_COMMS_MODEL's
        # own docstring in resolver_agent/crew/orchestrator.py).
        self.researcher_model = _litellm_model(researcher_model or os.environ.get("ANTHROPIC_MODEL_RESEARCHER") or model)
        self.decision_model = _litellm_model(decision_model or os.environ.get("ANTHROPIC_MODEL_DECISION") or model)
        self.comms_model = _litellm_model(comms_model or os.environ.get("ANTHROPIC_MODEL_COMMS") or DEFAULT_COMMS_MODEL)
        self.researcher_llm = LLM(model=self.researcher_model, max_tokens=_MAX_TOKENS)
        self.decision_llm = LLM(model=self.decision_model, max_tokens=_MAX_TOKENS)
        self.comms_llm = LLM(model=self.comms_model, max_tokens=_MAX_TOKENS)
        self.max_iterations_per_agent = max_iterations_per_agent

    def handle_ticket(self, ticket_text: str) -> CrewResult:
        case_id = uuid.uuid4().hex[:8]
        ctx = {"case_id": case_id}

        researcher_result = self._run_researcher(ticket_text, case_id)
        if researcher_result.report is None:
            log_event(
                _logger,
                logging.WARNING,
                "crew_agent.researcher_incomplete_escalating",
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
        decision_result = self._run_decision(risk_report, case_id)
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
                "crew_agent.decision_incomplete_escalating",
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
        comms_result = self._run_comms(decision, case_id)

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
        reasoning.extend(comms_result.warnings)
        reasoning.extend(f"[corrected] {c}" for c in comms_result.corrections)
        reasoning.append(f"Alert sent: {comms_result.alert_sent}.")

        log_event(
            _logger,
            logging.INFO,
            "crew_agent.case_resolved",
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

    # ----------------------------------------------------------------- #
    # Per-stage helpers -- each builds a fresh single-agent Crew per ticket
    # ----------------------------------------------------------------- #

    def _run_researcher(self, ticket_text: str, case_id: str) -> ResearcherResult:
        ctx = {"case_id": case_id, "agent_role": "researcher"}
        call_log: List[ToolCallRecord] = []
        tools = build_researcher_tools(call_log)

        agent = Agent(
            role=RESEARCHER_ROLE,
            goal=RESEARCHER_GOAL,
            backstory=RESEARCHER_BACKSTORY,
            tools=tools,
            llm=self.researcher_llm,
            max_iter=self.max_iterations_per_agent,
            verbose=False,
        )
        task = Task(
            description=RESEARCHER_TASK_DESCRIPTION.format(ticket_text=ticket_text),
            expected_output=RESEARCHER_TASK_EXPECTED_OUTPUT,
            agent=agent,
            output_pydantic=ResearcherOutput,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        try:
            crew.kickoff()
        except anthropic.APIError as exc:
            log_event(_logger, logging.ERROR, "crew_agent.researcher.api_error", error_type=type(exc).__name__, **ctx)
            return ResearcherResult(
                report=None,
                error={"error": "MODEL_API_ERROR", "message": str(exc)},
                tool_calls=call_log,
                stopped_reason="api_error",
            )
        except pydantic.ValidationError as exc:
            # CrewAI's own output_pydantic parsing can raise this raw (seen
            # live: the model's final answer occasionally collapses to just
            # one nested field, e.g. `evidence`, instead of the full
            # ResearcherOutput shape) rather than degrading gracefully --
            # equivalent to "the model never called submit_risk_report" in
            # the hand-rolled version, so it gets the same fallback, not a
            # crash.
            log_event(_logger, logging.WARNING, "crew_agent.researcher.output_parse_failed", detail=str(exc), **ctx)
            return ResearcherResult(
                report=None,
                error={"error": "NO_RISK_REPORT", "message": f"The researcher's output did not parse: {exc}"},
                tool_calls=call_log,
                stopped_reason="stop",
            )

        raw = self._extract_pydantic_dict(task)
        if raw is None:
            log_event(_logger, logging.WARNING, "crew_agent.researcher.no_report_produced", **ctx)
            return ResearcherResult(
                report=None,
                error={
                    "error": "NO_RISK_REPORT",
                    "message": "The researcher did not produce a structured risk report.",
                },
                tool_calls=call_log,
                stopped_reason="stop",
            )
        raw = _backfill_researcher_ok_fields(raw)

        errors = _validate_researcher_schema(raw)
        if errors:
            log_event(_logger, logging.WARNING, "crew_agent.researcher.invalid_report", errors=errors, **ctx)
            return ResearcherResult(
                report=None,
                error={"error": "INVALID_RISK_REPORT", "message": "; ".join(errors)},
                tool_calls=call_log,
                stopped_reason="stop",
            )

        if raw["status"] == "LOOKUP_FAILED":
            log_event(_logger, logging.INFO, "crew_agent.researcher.lookup_failed", error=raw.get("error"), **ctx)
            return ResearcherResult(
                report=None,
                error=raw.get("error"),
                order_id=raw.get("order_id"),
                tool_calls=call_log,
                stopped_reason="stop",
            )

        # status == "OK": cross-check against the real audit_fraud_risk
        # result before trusting a single field of it -- see
        # enforce_risk_report's own docstring for why.
        corrected, warnings, corrections = enforce_risk_report(raw, call_log)
        if corrections:
            log_event(_logger, logging.WARNING, "crew_agent.researcher.report_corrected", correction_count=len(corrections), **ctx)

        if corrected["status"] == "LOOKUP_FAILED":
            return ResearcherResult(
                report=None,
                error=corrected.get("error"),
                order_id=corrected.get("order_id"),
                warnings=warnings,
                corrections=corrections,
                tool_calls=call_log,
                stopped_reason="stop",
            )

        report = RiskReport(
            order_id=corrected["order_id"],
            user_id=corrected["user_id"],
            risk_score=corrected["risk_score"],
            risk_band=corrected["risk_band"],
            action_hint=corrected["action_hint"],
            triggered_rules=corrected["triggered_rules"],
            evidence=corrected["evidence"],
            blocks_automatic_refund=corrected["blocks_automatic_refund"],
            requires_security_channel=corrected["requires_security_channel"],
            rulebook_version=corrected["rulebook_version"],
        )
        log_event(
            _logger,
            logging.INFO,
            "crew_agent.researcher.report_produced",
            risk_band=report.risk_band,
            risk_score=report.risk_score,
            **ctx,
        )
        return ResearcherResult(
            report=report,
            error=None,
            order_id=report.order_id,
            warnings=warnings,
            corrections=corrections,
            tool_calls=call_log,
            stopped_reason="stop",
        )

    def _run_decision(self, risk_report: RiskReport, case_id: str) -> DecisionResult:
        ctx = {"case_id": case_id, "agent_role": "decision"}
        call_log: List[ToolCallRecord] = []
        tools = build_decision_tools(call_log)

        agent = Agent(
            role=DECISION_ROLE,
            goal=DECISION_GOAL,
            backstory=DECISION_BACKSTORY,
            tools=tools,
            llm=self.decision_llm,
            max_iter=self.max_iterations_per_agent,
            verbose=False,
        )
        task = Task(
            description=DECISION_TASK_DESCRIPTION.format(risk_report_json=risk_report.model_dump_json()),
            expected_output=DECISION_TASK_EXPECTED_OUTPUT,
            agent=agent,
            output_pydantic=DecisionOutput,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        try:
            crew.kickoff()
        except anthropic.APIError as exc:
            log_event(_logger, logging.ERROR, "crew_agent.decision.api_error", error_type=type(exc).__name__, **ctx)
            return DecisionResult(
                decision=None,
                error={"error": "MODEL_API_ERROR", "message": str(exc)},
                tool_calls=call_log,
                stopped_reason="api_error",
            )
        except pydantic.ValidationError as exc:
            # See _run_researcher's identical handling above -- CrewAI's own
            # output_pydantic parsing can raise this raw instead of
            # degrading gracefully.
            log_event(_logger, logging.WARNING, "crew_agent.decision.output_parse_failed", detail=str(exc), **ctx)
            return DecisionResult(
                decision=None,
                error={"error": "NO_DECISION", "message": f"The decision agent's output did not parse: {exc}"},
                tool_calls=call_log,
                stopped_reason="stop",
            )

        raw = self._extract_pydantic_dict(task)
        if raw is None:
            log_event(_logger, logging.WARNING, "crew_agent.decision.no_decision_produced", **ctx)
            return DecisionResult(
                decision=None,
                error={
                    "error": "NO_DECISION",
                    "message": "The decision agent did not produce a structured decision.",
                },
                tool_calls=call_log,
                stopped_reason="stop",
            )

        errors = _validate_decision_schema(raw)
        if errors:
            log_event(_logger, logging.WARNING, "crew_agent.decision.invalid_decision", errors=errors, **ctx)
            return DecisionResult(
                decision=None,
                error={"error": "INVALID_DECISION", "message": "; ".join(errors)},
                tool_calls=call_log,
                stopped_reason="stop",
            )

        corrected, warnings, corrections = enforce_decision(raw, call_log, risk_report)
        if corrections:
            log_event(_logger, logging.WARNING, "crew_agent.decision.corrected", correction_count=len(corrections), **ctx)

        decision = Decision(
            order_id=corrected["order_id"],
            user_id=corrected["user_id"],
            verdict=corrected["verdict"],
            eligible=corrected["eligible"],
            refund_status=corrected["refund_status"],
            requested_amount=corrected["requested_amount"],
            approved_amount=corrected.get("approved_amount"),
            refund_id=corrected.get("refund_id"),
            applicable_policies=corrected["applicable_policies"],
            rationale=corrected["rationale"],
            risk_report=risk_report,
        )
        log_event(_logger, logging.INFO, "crew_agent.decision.decision_produced", refund_status=decision.refund_status, **ctx)
        return DecisionResult(
            decision=decision,
            error=None,
            warnings=warnings,
            corrections=corrections,
            tool_calls=call_log,
            stopped_reason="stop",
        )

    def _run_comms(self, decision: Decision, case_id: str) -> CommsResult:
        ctx = {"case_id": case_id, "agent_role": "comms"}
        call_log: List[ToolCallRecord] = []
        tools = build_comms_tools(call_log, case_id)

        agent = Agent(
            role=COMMS_ROLE,
            goal=COMMS_GOAL,
            backstory=COMMS_BACKSTORY,
            tools=tools,
            llm=self.comms_llm,
            max_iter=self.max_iterations_per_agent,
            verbose=False,
        )
        # output_pydantic=CommsOutput even for a single field -- taking the
        # raw final-answer text directly (tried first) let the model's own
        # preamble/scratchpad leak into customer_response (observed live,
        # e.g. "I'll now write the customer reply: ---\n\n..."); forcing a
        # one-field schema makes the model isolate exactly that field.
        task = Task(
            description=COMMS_TASK_DESCRIPTION.format(decision_json=decision.model_dump_json()),
            expected_output=COMMS_TASK_EXPECTED_OUTPUT,
            agent=agent,
            output_pydantic=CommsOutput,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        try:
            crew.kickoff()
        except (anthropic.APIError, pydantic.ValidationError) as exc:
            log_event(_logger, logging.ERROR, "crew_agent.comms.api_error", error_type=type(exc).__name__, **ctx)
            return CommsResult(
                customer_response=_safe_customer_response(decision.refund_status),
                escalation=None,
                alert_sent=False,
                alert_record=None,
                tool_calls=call_log,
                stopped_reason="api_error" if isinstance(exc, anthropic.APIError) else "stop",
            )

        escalation = CommsAgent._last_successful(call_log, "get_escalation_route")
        alert_record = CommsAgent._last_successful(call_log, "send_slack_alert", require_key="delivered")
        alert_sent = alert_record is not None

        if escalation is not None and escalation.get("escalation_required") and not alert_sent:
            log_event(
                _logger,
                logging.WARNING,
                "crew_agent.comms.escalation_required_but_no_alert_sent",
                channel_id=escalation.get("channel_id"),
                **ctx,
            )

        raw = self._extract_pydantic_dict(task)
        customer_response = ((raw or {}).get("customer_response") or "").strip()
        warnings: List[str] = []
        corrections: List[str] = []
        if not customer_response:
            log_event(_logger, logging.WARNING, "crew_agent.comms.fallback_customer_response", **ctx)
            customer_response = _safe_customer_response(decision.refund_status)
        else:
            # Two independent, deterministic checks on the same underlying
            # rule -- see comms/output_tool.py's own module docstring.
            violation = find_stale_refund_detail(customer_response, decision) or find_premature_approval_language(
                customer_response, decision
            )
            if violation is not None:
                warnings.append(violation)
                log_event(_logger, logging.WARNING, "crew_agent.comms.customer_response_overstated", detail=violation, **ctx)
                customer_response = _safe_customer_response(decision.refund_status)
                corrections.append(f"customer_response replaced with a safe generic reply: {violation}")

        log_event(_logger, logging.INFO, "crew_agent.comms.result_produced", alert_sent=alert_sent, **ctx)
        return CommsResult(
            customer_response=customer_response,
            escalation=escalation,
            alert_sent=alert_sent,
            alert_record=alert_record,
            tool_calls=call_log,
            stopped_reason="stop",
            warnings=warnings,
            corrections=corrections,
        )

    @staticmethod
    def _extract_pydantic_dict(task: Task) -> Optional[Dict[str, Any]]:
        output = task.output
        if output is None or output.pydantic is None:
            return None
        return output.pydantic.model_dump()
