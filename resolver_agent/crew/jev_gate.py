"""A Jev-scored triage gate inserted between Agent 1 (Researcher) and Agent 2
(Decision Maker).

Grounds *before* scoring -- never scores raw ticket text alone. The
Researcher's own tool calls already ground the fraud/customer picture
(``RiskReport``); this module adds one more grounded fact
(``check_return_policy``'s verdict for this order) before ever calling Jev,
then lets the score gate whether ``DecisionAgent`` needs to run its normal
freeform reasoning at all.

Every branch -- fast or not -- still produces raw ``submit_decision``
arguments that pass through the *exact same* :func:`enforce_decision` safety
net orchestrator.py already uses today (including the ORD-1005 fraud-block
guardrail). This module adds no new safety logic of its own; it only decides
*how* a decision gets produced, never trusts one without that existing
cross-check.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import anthropic

from ..jev_client import JevClient, JevScore
from ..logging_utils import get_logger, log_event
from ..tool_loop import ModelAPIError, ToolCallRecord, ToolLoopResult, run_tool_loop, signature
from .decision.agent import DecisionResult
from .decision.output_tool import SUBMIT_DECISION_SCHEMA, SUBMIT_DECISION_TOOL_NAME, enforce_decision, validate_schema
from .decision.prompts import DECISION_PROMPT
from .schemas import Decision, RiskReport

_logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Config -- env-var overridable, same pattern as agent.py's DEFAULT_MODEL.
# --------------------------------------------------------------------------- #

JEV_GATE_ENABLED = os.environ.get("JEV_GATE_ENABLED", "false").lower() == "true"
JEV_APPROVE_SCORE_THRESHOLD = float(os.environ.get("JEV_APPROVE_SCORE_THRESHOLD", "0.85"))
JEV_REJECT_SCORE_THRESHOLD = float(os.environ.get("JEV_REJECT_SCORE_THRESHOLD", "0.15"))
JEV_MIN_CONFIDENCE = float(os.environ.get("JEV_MIN_CONFIDENCE", "0.90"))

# --------------------------------------------------------------------------- #
# Step A -- extract {reason, claim_summary, ...} from ticket_text alone.
#
# RiskReport carries no "what is the customer claiming" field -- Researcher's
# tools are order/user/fraud lookups, not ticket-reading. Only reading the
# ticket text itself can produce the `reason` check_return_policy needs.
# --------------------------------------------------------------------------- #

TRIAGE_QUERY_TOOL_NAME = "extract_triage_query"

_REASON_VALUES = ["damaged_on_arrival", "wrong_item", "item_missing", "late_delivery", "changed_mind"]

TRIAGE_QUERY_SCHEMA: Dict[str, Any] = {
    "name": TRIAGE_QUERY_TOOL_NAME,
    "description": (
        "Record a structured summary of what this customer is claiming, "
        "read only from the ticket text. Call this exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": _REASON_VALUES,
                "description": "The customer's claimed reason for a return/refund -- the closest match to what the ticket says.",
            },
            "claim_summary": {
                "type": "string",
                "description": "A 1-2 sentence paraphrase of the claim. Do not invent specifics not in the ticket.",
            },
            "customer_stated_amount": {
                "type": ["number", "null"],
                "description": "An amount the customer explicitly asked for, in USD, or null if none was stated.",
            },
            "sentiment": {
                "type": "string",
                "enum": ["calm", "frustrated", "angry", "anxious"],
                "description": "Based only on what the ticket text actually shows, never invented.",
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": ["reason", "claim_summary", "customer_stated_amount", "sentiment", "urgency"],
    },
}

EXTRACTION_PROMPT = """\
You are reading one GlobalCart support ticket to summarize what the \
customer is claiming -- you have no tools to look anything up, and nothing \
you write here is shown to the customer. Read the ticket text and call \
extract_triage_query exactly once, basing every field only on what the \
ticket actually says, never inventing a detail it doesn't contain. If \
several reasons could apply, pick the single closest match.
"""


def validate_triage_query_schema(extracted: Dict[str, Any]) -> List[str]:
    """Structural/type completeness check, independent of the API's own
    tool-schema enforcement -- same recipe as output_tool.py's
    validate_schema functions elsewhere in this codebase."""
    errors: List[str] = []

    if extracted.get("reason") not in _REASON_VALUES:
        errors.append(f"reason is missing or not one of {_REASON_VALUES}: got {extracted.get('reason')!r}.")
    if not isinstance(extracted.get("claim_summary"), str) or not extracted["claim_summary"].strip():
        errors.append("claim_summary is missing, not a string, or empty.")
    amount = extracted.get("customer_stated_amount")
    if amount is not None and (isinstance(amount, bool) or not isinstance(amount, (int, float))):
        errors.append("customer_stated_amount must be a number or null.")
    if extracted.get("sentiment") not in ("calm", "frustrated", "angry", "anxious"):
        errors.append(f"sentiment is missing or invalid: got {extracted.get('sentiment')!r}.")
    if extracted.get("urgency") not in ("low", "medium", "high"):
        errors.append(f"urgency is missing or invalid: got {extracted.get('urgency')!r}.")

    return errors


# --------------------------------------------------------------------------- #
# Gate outcome + orchestration
# --------------------------------------------------------------------------- #


@dataclass
class GateOutcome:
    """What one :func:`run_gate` call produces.

    ``decision_result`` is set (and authoritative) for ``fast_approve``/
    ``fast_reject`` -- the caller should use it directly instead of running
    ``DecisionAgent``. For ``fallthrough``, ``seed_messages``/
    ``seed_seen_calls`` carry whatever grounding already happened (possibly
    none) for ``DecisionAgent.run`` to build on instead of repeating it.
    """

    branch: str  # "fast_approve" | "fast_reject" | "fallthrough"
    decision_result: Optional[DecisionResult] = None
    seed_messages: Optional[List[Dict[str, Any]]] = None
    seed_seen_calls: Optional[set] = None
    grounding_tool_calls: List[ToolCallRecord] = field(default_factory=list)


def _stringify(result: Any) -> str:
    try:
        return json.dumps(result)
    except TypeError:
        return str(result)


def _tool_call_messages(calls: List[ToolCallRecord]) -> List[Dict[str, Any]]:
    """Turn already-executed tool calls into the assistant/tool_result
    message pairs a fresh ``run_tool_loop`` call can be seeded with, so the
    model sees real prior tool results without this loop re-executing them.
    """
    messages: List[Dict[str, Any]] = []
    for i, call in enumerate(calls):
        tool_use_id = f"jev_gate_{i}"
        messages.append(
            {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use_id, "name": call.name, "input": call.input}]}
        )
        messages.append(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": _stringify(call.result)}]}
        )
    return messages


def _extract_triage_query(
    client: "anthropic.Anthropic", model: str, ticket_text: str, ctx: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    try:
        result = run_tool_loop(
            client=client,
            model=model,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": ticket_text}],
            tool_schemas=[TRIAGE_QUERY_SCHEMA],
            tool_registry={},
            stop_tool_name=TRIAGE_QUERY_TOOL_NAME,
            max_iterations=1,
            log_context=ctx,
        )
    except ModelAPIError:
        log_event(_logger, logging.WARNING, "jev_gate.extraction_api_error", **ctx)
        return None

    raw = None
    for call in reversed(result.tool_calls):
        if call.name == TRIAGE_QUERY_TOOL_NAME:
            raw = dict(call.input)
            break
    if raw is None:
        log_event(_logger, logging.WARNING, "jev_gate.extraction_invalid", reason="no_call", **ctx)
        return None

    errors = validate_triage_query_schema(raw)
    if errors:
        log_event(_logger, logging.WARNING, "jev_gate.extraction_invalid", errors=errors, **ctx)
        return None
    return raw


def _forced_decision_call(
    client: "anthropic.Anthropic",
    model: str,
    seed_messages: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> ToolLoopResult:
    return run_tool_loop(
        client=client,
        model=model,
        system=DECISION_PROMPT,
        messages=seed_messages,
        tool_schemas=[SUBMIT_DECISION_SCHEMA],
        tool_registry={},
        stop_tool_name=SUBMIT_DECISION_TOOL_NAME,
        max_iterations=1,
        log_context=ctx,
    )


def _extract_decision_input(tool_calls: List[ToolCallRecord]) -> Optional[Dict[str, Any]]:
    for call in reversed(tool_calls):
        if call.name == SUBMIT_DECISION_TOOL_NAME:
            return dict(call.input)
    return None


def _finish_fast_path(
    *,
    client: "anthropic.Anthropic",
    model: str,
    risk_report: RiskReport,
    grounding_calls: List[ToolCallRecord],
    ctx: Dict[str, Any],
) -> DecisionResult:
    """Shared tail for fast_approve/fast_reject: one forced submit_decision
    call seeded with whatever grounding already happened, then the exact
    same validate_schema + enforce_decision safety net DecisionAgent uses.
    """
    seed_messages = [{"role": "user", "content": risk_report.model_dump_json()}] + _tool_call_messages(grounding_calls)

    try:
        result = _forced_decision_call(client, model, seed_messages, ctx)
    except ModelAPIError as exc:
        log_event(_logger, logging.ERROR, "jev_gate.fast_path_api_error", **ctx)
        return DecisionResult(
            decision=None,
            error={"error": "MODEL_API_ERROR", "message": str(exc)},
            tool_calls=grounding_calls + exc.tool_calls,
            stopped_reason="api_error",
        )

    all_tool_calls = grounding_calls + result.tool_calls
    raw = _extract_decision_input(result.tool_calls)
    if raw is None:
        log_event(_logger, logging.WARNING, "jev_gate.fast_path_no_decision", stopped_reason=result.stopped_reason, **ctx)
        return DecisionResult(
            decision=None,
            error={
                "error": "NO_DECISION",
                "message": f"The gate's forced submit_decision call did not produce one (stopped_reason={result.stopped_reason!r}).",
            },
            tool_calls=all_tool_calls,
            stopped_reason=result.stopped_reason,
        )

    errors = validate_schema(raw)
    if errors:
        log_event(_logger, logging.WARNING, "jev_gate.fast_path_invalid_decision", errors=errors, **ctx)
        return DecisionResult(
            decision=None,
            error={"error": "INVALID_DECISION", "message": "; ".join(errors)},
            tool_calls=all_tool_calls,
            stopped_reason=result.stopped_reason,
        )

    corrected, warnings, corrections = enforce_decision(raw, all_tool_calls, risk_report)
    if corrections:
        log_event(_logger, logging.WARNING, "jev_gate.fast_path_corrected", correction_count=len(corrections), **ctx)

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
    return DecisionResult(
        decision=decision,
        error=None,
        warnings=warnings,
        corrections=corrections,
        tool_calls=all_tool_calls,
        stopped_reason=result.stopped_reason,
    )


def run_gate(
    *,
    client: "anthropic.Anthropic",
    model: str,
    ticket_text: str,
    risk_report: RiskReport,
    tool_registry: Dict[str, Callable[..., Any]],
    jev_client: JevClient,
    case_id: str,
) -> GateOutcome:
    """Ground (check_return_policy), score (Jev), and gate what happens next.

    Never raises for a Jev/extraction/grounding failure -- every failure
    mode degrades to ``fallthrough`` (today's unchanged DecisionAgent
    behavior, seeded with whatever grounding did succeed), logged via
    ``jev_gate.*`` events. Only ``ModelAPIError`` from the fast paths'
    forced final call propagates information back via
    ``DecisionResult.error`` (mirroring how the rest of this codebase
    surfaces API failures), never as an exception the orchestrator has to
    catch specially.
    """
    ctx = {"case_id": case_id, "agent_role": "jev_gate"}

    extracted = _extract_triage_query(client, model, ticket_text, ctx)
    if extracted is None:
        return GateOutcome(branch="fallthrough")

    check_fn = tool_registry["check_return_policy"]
    policy_result = check_fn(order_id=risk_report.order_id, reason=extracted["reason"])
    policy_call = ToolCallRecord(
        "check_return_policy", {"order_id": risk_report.order_id, "reason": extracted["reason"]}, policy_result
    )

    if isinstance(policy_result, dict) and "error" in policy_result:
        log_event(_logger, logging.WARNING, "jev_gate.grounding_error", error=policy_result.get("error"), **ctx)
        return GateOutcome(branch="fallthrough")

    query = {
        "risk_report": {
            "risk_score": risk_report.risk_score,
            "risk_band": risk_report.risk_band,
            "blocks_automatic_refund": risk_report.blocks_automatic_refund,
        },
        "check_return_policy": {
            "eligible": policy_result.get("eligible"),
            "verdict": policy_result.get("verdict"),
            "requires_escalation": policy_result.get("requires_escalation"),
            "max_refundable_amount": policy_result.get("max_refundable_amount"),
        },
        "ticket": {
            "claim_summary": extracted["claim_summary"],
            "customer_stated_amount": extracted["customer_stated_amount"],
            "sentiment": extracted["sentiment"],
            "urgency": extracted["urgency"],
        },
    }

    seed_seen = {signature("check_return_policy", policy_call.input)}

    try:
        jev_score: JevScore = jev_client.score(query)
    except Exception:  # noqa: BLE001 -- any Jev failure degrades to fallthrough, never a hard error.
        log_event(_logger, logging.WARNING, "jev_gate.client_error", **ctx)
        return GateOutcome(
            branch="fallthrough",
            seed_messages=[{"role": "user", "content": risk_report.model_dump_json()}] + _tool_call_messages([policy_call]),
            seed_seen_calls=seed_seen,
            grounding_tool_calls=[policy_call],
        )

    log_event(
        _logger,
        logging.INFO,
        "jev_gate.score_computed",
        score=jev_score.score,
        confidence=jev_score.confidence,
        **ctx,
    )

    branch = _classify(jev_score, policy_result, risk_report)
    log_event(_logger, logging.INFO, "jev_gate.branch_selected", branch=branch, **ctx)

    if branch == "fallthrough":
        return GateOutcome(
            branch="fallthrough",
            seed_messages=[{"role": "user", "content": risk_report.model_dump_json()}] + _tool_call_messages([policy_call]),
            seed_seen_calls=seed_seen,
            grounding_tool_calls=[policy_call],
        )

    if branch == "fast_reject":
        decision_result = _finish_fast_path(
            client=client, model=model, risk_report=risk_report, grounding_calls=[policy_call], ctx=ctx
        )
        return GateOutcome(branch="fast_reject", decision_result=decision_result, grounding_tool_calls=[policy_call])

    # fast_approve -- process_refund is called for real; its own status
    # (APPROVED/REJECTED/ESCALATION_REQUIRED) is the ground truth
    # enforce_decision cross-checks against, regardless of what this
    # branch's own label assumed.
    refund_fn = tool_registry["process_refund"]
    amount = policy_result["max_refundable_amount"]
    refund_result = refund_fn(order_id=risk_report.order_id, amount=amount, reason=extracted["reason"])
    refund_call = ToolCallRecord(
        "process_refund", {"order_id": risk_report.order_id, "amount": amount, "reason": extracted["reason"]}, refund_result
    )
    decision_result = _finish_fast_path(
        client=client, model=model, risk_report=risk_report, grounding_calls=[policy_call, refund_call], ctx=ctx
    )
    return GateOutcome(branch="fast_approve", decision_result=decision_result, grounding_tool_calls=[policy_call, refund_call])


def _classify(jev_score: JevScore, policy_result: Dict[str, Any], risk_report: RiskReport) -> str:
    """Score gates the branch; grounded facts gate whether that branch is
    even attemptable -- e.g. a high score on an ineligible order (or one
    with a $0 max_refundable_amount) must never reach fast_approve, since
    process_refund would reject a $0/invalid amount outright. Same for a
    report with blocks_automatic_refund=True: enforce_decision would still
    correct a wrongly-fast-approved case downstream (the ORD-1005
    guardrail), but there is no reason to attempt process_refund and a
    forced write-up call at all on a case already known to be fraud-blocked.
    """
    if jev_score.confidence < JEV_MIN_CONFIDENCE:
        return "fallthrough"

    eligible = policy_result.get("eligible")
    if (
        jev_score.score >= JEV_APPROVE_SCORE_THRESHOLD
        and eligible is True
        and not policy_result.get("requires_escalation")
        and not risk_report.blocks_automatic_refund
        and (policy_result.get("max_refundable_amount") or 0) > 0
    ):
        return "fast_approve"

    if jev_score.score <= JEV_REJECT_SCORE_THRESHOLD and eligible is False:
        return "fast_reject"

    return "fallthrough"
