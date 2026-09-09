"""ResearcherAgent -- wires multi_agent_tools.RESEARCHER_TOOLS and the
submit_risk_report handoff into the generic tool_loop.

Structurally identical to resolver_agent/agent.py's ResolverAgent: build a
tool list from a fixed bundle plus one handoff tool, run the generic loop,
extract and validate the forced final call. tool_loop.py itself is reused
completely unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

import multi_agent_tools as mat  # noqa: E402  (starter-kit/ is on sys.path -- see crew/__init__.py)

from ...logging_utils import get_logger, log_event
from ...tool_loop import ModelAPIError, ToolCallRecord, run_tool_loop
from ..schemas import RiskReport
from .output_tool import (
    SUBMIT_RISK_REPORT_SCHEMA,
    SUBMIT_RISK_REPORT_TOOL_NAME,
    enforce_risk_report,
    validate_schema,
)
from .prompts import RESEARCHER_PROMPT

_logger = get_logger(__name__)

_RESEARCHER_TOOL_NAMES = ("get_order_details", "get_user_profile", "audit_fraud_risk")


@dataclass
class ResearcherResult:
    """What one ResearcherAgent.run() call produces.

    Exactly one of ``report``/``error`` is meaningful at a time: a clean run
    sets ``report`` and leaves ``error`` None; a lookup failure (or a
    structurally invalid / missing submit_risk_report call, treated the same
    way resolver_agent.agent._fallback_resolution treats a bad
    submit_resolution call) sets ``error`` and leaves ``report`` None. The
    orchestrator's job is simply: no report -> escalate, never re-dispatch.
    """

    report: Optional[RiskReport]
    error: Optional[Dict[str, Any]]
    order_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    stopped_reason: str = "stop"


class ResearcherAgent:
    """Agent 1 -- investigates one ticket and produces a RiskReport. Cannot
    approve money or message the customer; its tool_registry physically
    only contains the three RESEARCHER_TOOLS names."""

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
        self.tool_schemas = list(mat.RESEARCHER_TOOLS) + [SUBMIT_RISK_REPORT_SCHEMA]
        self.tool_registry = {name: mat.TOOL_REGISTRY[name] for name in _RESEARCHER_TOOL_NAMES}

    def run(self, ticket_text: str, case_id: str) -> ResearcherResult:
        ctx = {"case_id": case_id, "agent_role": "researcher"}
        messages: List[Dict[str, Any]] = [{"role": "user", "content": ticket_text}]

        try:
            result = run_tool_loop(
                client=self.client,
                model=self.model,
                system=RESEARCHER_PROMPT,
                messages=messages,
                tool_schemas=self.tool_schemas,
                tool_registry=self.tool_registry,
                stop_tool_name=SUBMIT_RISK_REPORT_TOOL_NAME,
                max_iterations=self.max_iterations,
                log_context=ctx,
            )
        except ModelAPIError as exc:
            log_event(
                _logger,
                logging.ERROR,
                "researcher.api_error",
                error_type=type(exc.original).__name__,
                **ctx,
            )
            return ResearcherResult(
                report=None,
                error={"error": "MODEL_API_ERROR", "message": str(exc)},
                tool_calls=exc.tool_calls,
                stopped_reason="api_error",
            )

        raw = self._extract_report(result.tool_calls)
        if raw is None:
            log_event(_logger, logging.WARNING, "researcher.no_report_produced", stopped_reason=result.stopped_reason, **ctx)
            return ResearcherResult(
                report=None,
                error={
                    "error": "NO_RISK_REPORT",
                    "message": f"The researcher did not call submit_risk_report (stopped_reason={result.stopped_reason!r}).",
                },
                tool_calls=result.tool_calls,
                stopped_reason=result.stopped_reason,
            )

        errors = validate_schema(raw)
        if errors:
            log_event(_logger, logging.WARNING, "researcher.invalid_report", errors=errors, **ctx)
            return ResearcherResult(
                report=None,
                error={"error": "INVALID_RISK_REPORT", "message": "; ".join(errors)},
                tool_calls=result.tool_calls,
                stopped_reason=result.stopped_reason,
            )

        if raw["status"] == "LOOKUP_FAILED":
            log_event(_logger, logging.INFO, "researcher.lookup_failed", error=raw.get("error"), **ctx)
            return ResearcherResult(
                report=None,
                error=raw.get("error"),
                order_id=raw.get("order_id"),
                tool_calls=result.tool_calls,
                stopped_reason=result.stopped_reason,
            )

        # status == "OK": cross-check against the real audit_fraud_risk
        # result before trusting a single field of it -- see
        # output_tool.enforce_risk_report's docstring for why this can't be
        # left to the prompt alone.
        corrected, warnings, corrections = enforce_risk_report(raw, result.tool_calls)
        if corrections:
            log_event(_logger, logging.WARNING, "researcher.report_corrected", correction_count=len(corrections), **ctx)

        if corrected["status"] == "LOOKUP_FAILED":
            return ResearcherResult(
                report=None,
                error=corrected.get("error"),
                order_id=corrected.get("order_id"),
                warnings=warnings,
                corrections=corrections,
                tool_calls=result.tool_calls,
                stopped_reason=result.stopped_reason,
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
            "researcher.report_produced",
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
            tool_calls=result.tool_calls,
            stopped_reason=result.stopped_reason,
        )

    @staticmethod
    def _extract_report(tool_calls: List[ToolCallRecord]) -> Optional[Dict[str, Any]]:
        for call in reversed(tool_calls):
            if call.name == SUBMIT_RISK_REPORT_TOOL_NAME:
                return dict(call.input)
        return None
