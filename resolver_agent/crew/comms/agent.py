"""CommsAgent -- wires multi_agent_tools.COMMS_TOOLS and the
submit_comms_result handoff into the generic tool_loop.

Same shape as the other two crew agents. The one thing unique to this agent:
a dispatch-boundary guardrail on send_slack_alert, built fresh per ``run()``
call (never shared across cases, same reasoning as
resolver_agent.agent._authorize_tool_registry returning a new dict per call
rather than mutating self.tool_registry) -- a real Slack/outbox write can
only happen after this same case's own get_escalation_route call actually
returned escalation_required=true. This is enforced in code, not only in the
prompt, exactly the way resolver_agent's own guardrails are: a prompt
instruction is a suggestion, a dispatch-boundary check is a guarantee.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import anthropic

import multi_agent_tools as mat  # noqa: E402  (starter-kit/ is on sys.path -- see crew/__init__.py)

from ...logging_utils import get_logger, log_event
from ...tool_loop import ModelAPIError, ToolCallRecord, run_tool_loop
from ..schemas import Decision
from .output_tool import SUBMIT_COMMS_RESULT_SCHEMA, SUBMIT_COMMS_RESULT_TOOL_NAME, validate_schema
from .prompts import COMMS_PROMPT

_logger = get_logger(__name__)

_COMMS_TOOL_NAMES = ("get_escalation_route", "send_slack_alert")

_ALERT_DENIED = {
    "error": "ALERT_NOT_AUTHORIZED",
    "message": (
        "send_slack_alert was refused: this case's own get_escalation_route "
        "call has not returned escalation_required=true. Do not call "
        "send_slack_alert on a case that resolved cleanly."
    ),
}


def _safe_customer_response(refund_status: Optional[str]) -> str:
    """A safe, generic reply for when the model's own customer_response is
    missing or invalid -- mirrors resolver_agent.output_tool's
    _safe_customer_response, scoped to this crew's 3-value refund_status."""
    if refund_status == "APPROVED":
        return "Good news -- your refund has been approved and is on its way."
    if refund_status == "REJECTED":
        return "After reviewing your case, we're not able to process a refund for this order."
    return (
        "This case needs a closer look from our operations team before we "
        "can give you a final answer -- we're looking into it now and will "
        "follow up shortly."
    )


def _guarded_registry(base_registry: Dict[str, Callable[..., Any]], case_id: str) -> Dict[str, Callable[..., Any]]:
    """Wrap ``send_slack_alert`` so it only ever dispatches for real after
    this same case's own ``get_escalation_route`` call returned
    escalation_required=true. ``state`` is local to one call to
    :meth:`CommsAgent.run` -- a fresh dict every time, so one case's routing
    result can never leak into another's alert decision.
    """
    state = {"escalation_required": False}
    real_route = base_registry["get_escalation_route"]
    real_alert = base_registry["send_slack_alert"]

    def route(**kwargs: Any) -> Any:
        result = real_route(**kwargs)
        if isinstance(result, dict) and "error" not in result:
            state["escalation_required"] = bool(result.get("escalation_required"))
        return result

    def alert(**kwargs: Any) -> Any:
        if not state["escalation_required"]:
            log_event(_logger, logging.WARNING, "comms.alert_denied_no_escalation", case_id=case_id)
            return dict(_ALERT_DENIED)
        return real_alert(**kwargs)

    return {**base_registry, "get_escalation_route": route, "send_slack_alert": alert}


@dataclass
class CommsResult:
    """What one CommsAgent.run() call produces. ``escalation``/``alert_sent``/
    ``alert_record`` are read from the real tool_calls, never from the
    model's own claim -- only ``customer_response`` is the model's to write.
    """

    customer_response: str
    escalation: Optional[Dict[str, Any]]
    alert_sent: bool
    alert_record: Optional[Dict[str, Any]]
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    stopped_reason: str = "stop"


class CommsAgent:
    """Agent 3 -- routes and notifies, and drafts the customer reply. Cannot
    approve money; its tool_registry physically only contains
    get_escalation_route and send_slack_alert."""

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
        self.tool_schemas = list(mat.COMMS_TOOLS) + [SUBMIT_COMMS_RESULT_SCHEMA]
        self._base_tool_registry = {name: mat.TOOL_REGISTRY[name] for name in _COMMS_TOOL_NAMES}

    def run(self, decision: Decision, case_id: str) -> CommsResult:
        ctx = {"case_id": case_id, "agent_role": "comms"}
        messages: List[Dict[str, Any]] = [{"role": "user", "content": decision.model_dump_json()}]
        tool_registry = _guarded_registry(self._base_tool_registry, case_id)

        try:
            result = run_tool_loop(
                client=self.client,
                model=self.model,
                system=COMMS_PROMPT,
                messages=messages,
                tool_schemas=self.tool_schemas,
                tool_registry=tool_registry,
                stop_tool_name=SUBMIT_COMMS_RESULT_TOOL_NAME,
                max_iterations=self.max_iterations,
                log_context=ctx,
            )
        except ModelAPIError as exc:
            log_event(_logger, logging.ERROR, "comms.api_error", error_type=type(exc.original).__name__, **ctx)
            return CommsResult(
                customer_response=_safe_customer_response(decision.refund_status),
                escalation=None,
                alert_sent=False,
                alert_record=None,
                tool_calls=exc.tool_calls,
                stopped_reason="api_error",
            )

        escalation = self._last_successful(result.tool_calls, "get_escalation_route")
        alert_record = self._last_successful(result.tool_calls, "send_slack_alert", require_key="delivered")
        alert_sent = alert_record is not None

        if escalation is not None and escalation.get("escalation_required") and not alert_sent:
            log_event(
                _logger,
                logging.WARNING,
                "comms.escalation_required_but_no_alert_sent",
                channel_id=escalation.get("channel_id"),
                **ctx,
            )

        raw = self._extract_result(result.tool_calls)
        customer_response: str
        if raw is None or validate_schema(raw):
            log_event(_logger, logging.WARNING, "comms.fallback_customer_response", stopped_reason=result.stopped_reason, **ctx)
            customer_response = _safe_customer_response(decision.refund_status)
        else:
            customer_response = raw["customer_response"]

        log_event(_logger, logging.INFO, "comms.result_produced", alert_sent=alert_sent, **ctx)
        return CommsResult(
            customer_response=customer_response,
            escalation=escalation,
            alert_sent=alert_sent,
            alert_record=alert_record,
            tool_calls=result.tool_calls,
            stopped_reason=result.stopped_reason,
        )

    @staticmethod
    def _extract_result(tool_calls: List[ToolCallRecord]) -> Optional[Dict[str, Any]]:
        for call in reversed(tool_calls):
            if call.name == SUBMIT_COMMS_RESULT_TOOL_NAME:
                return dict(call.input)
        return None

    @staticmethod
    def _last_successful(
        tool_calls: List[ToolCallRecord], name: str, *, require_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for call in reversed(tool_calls):
            if call.name != name or not isinstance(call.result, dict):
                continue
            if "error" in call.result:
                continue
            if require_key is not None and not call.result.get(require_key):
                continue
            return call.result
        return None
