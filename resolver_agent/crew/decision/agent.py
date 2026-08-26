"""DecisionAgent -- wires multi_agent_tools.DECISION_TOOLS and the
submit_decision handoff into the generic tool_loop.

Same shape as resolver_agent/agent.py's ResolverAgent and
crew/researcher/agent.py's ResearcherAgent -- only the tool bundle, prompt,
and handoff schema differ. tool_loop.py is reused completely unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

import multi_agent_tools as mat  # noqa: E402  (starter-kit/ is on sys.path -- see crew/__init__.py)

from ...logging_utils import get_logger, log_event
from ...tool_loop import ModelAPIError, ToolCallRecord, run_tool_loop
from ..schemas import Decision, RiskReport
from .output_tool import SUBMIT_DECISION_SCHEMA, SUBMIT_DECISION_TOOL_NAME, enforce_decision, validate_schema
from .prompts import DECISION_PROMPT

_logger = get_logger(__name__)

_DECISION_TOOL_NAMES = ("check_return_policy", "process_refund")


@dataclass
class DecisionResult:
    """What one DecisionAgent.run() call produces. Same "exactly one of
    decision/error is meaningful" contract as
    crew.researcher.agent.ResearcherResult."""

    decision: Optional[Decision]
    error: Optional[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    stopped_reason: str = "stop"


class DecisionAgent:
    """Agent 2 -- consults policy and decides the financial outcome. Cannot
    investigate from scratch (no order/user lookup tools) and cannot talk to
    the customer; its tool_registry physically only contains
    check_return_policy and process_refund."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_iterations: int = 6,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations!r}.")
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.tool_schemas = list(mat.DECISION_TOOLS) + [SUBMIT_DECISION_SCHEMA]
        self.tool_registry = {name: mat.TOOL_REGISTRY[name] for name in _DECISION_TOOL_NAMES}

    def run(self, risk_report: RiskReport, case_id: str) -> DecisionResult:
        ctx = {"case_id": case_id, "agent_role": "decision"}
        messages: List[Dict[str, Any]] = [{"role": "user", "content": risk_report.model_dump_json()}]

        try:
            result = run_tool_loop(
                client=self.client,
                model=self.model,
                system=DECISION_PROMPT,
                messages=messages,
                tool_schemas=self.tool_schemas,
                tool_registry=self.tool_registry,
                stop_tool_name=SUBMIT_DECISION_TOOL_NAME,
                max_iterations=self.max_iterations,
                log_context=ctx,
            )
        except ModelAPIError as exc:
            log_event(_logger, logging.ERROR, "decision.api_error", error_type=type(exc.original).__name__, **ctx)
            return DecisionResult(
                decision=None,
                error={"error": "MODEL_API_ERROR", "message": str(exc)},
                tool_calls=exc.tool_calls,
                stopped_reason="api_error",
            )

        raw = self._extract_decision(result.tool_calls)
        if raw is None:
            log_event(_logger, logging.WARNING, "decision.no_decision_produced", stopped_reason=result.stopped_reason, **ctx)
            return DecisionResult(
                decision=None,
                error={
                    "error": "NO_DECISION",
                    "message": f"The decision agent did not call submit_decision (stopped_reason={result.stopped_reason!r}).",
                },
                tool_calls=result.tool_calls,
                stopped_reason=result.stopped_reason,
            )

        errors = validate_schema(raw)
        if errors:
            log_event(_logger, logging.WARNING, "decision.invalid_decision", errors=errors, **ctx)
            return DecisionResult(
                decision=None,
                error={"error": "INVALID_DECISION", "message": "; ".join(errors)},
                tool_calls=result.tool_calls,
                stopped_reason=result.stopped_reason,
            )

        corrected, warnings, corrections = enforce_decision(raw, result.tool_calls, risk_report)
        if corrections:
            log_event(_logger, logging.WARNING, "decision.corrected", correction_count=len(corrections), **ctx)

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
        log_event(_logger, logging.INFO, "decision.decision_produced", refund_status=decision.refund_status, **ctx)
        return DecisionResult(
            decision=decision,
            error=None,
            warnings=warnings,
            corrections=corrections,
            tool_calls=result.tool_calls,
            stopped_reason=result.stopped_reason,
        )

    @staticmethod
    def _extract_decision(tool_calls: List[ToolCallRecord]) -> Optional[Dict[str, Any]]:
        for call in reversed(tool_calls):
            if call.name == SUBMIT_DECISION_TOOL_NAME:
                return dict(call.input)
        return None
