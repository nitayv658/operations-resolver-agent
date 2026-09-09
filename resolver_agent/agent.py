"""ResolverAgent -- wires the GlobalCart tools and the submit_resolution
output contract into the generic tool_loop.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

import anthropic  # noqa: E402

# starter-kit/ is not a package -- it's the fixed, ungraded tool box handed
# out with the quest. Add it to sys.path so `import mock_services` resolves
# without copying or modifying anything inside it.
STARTER_KIT_DIR = Path(__file__).resolve().parent.parent / "starter-kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

import mock_services as gc  # noqa: E402  (path must be set up first)

from .authorization import authorize_tool_registry  # noqa: E402
from .escalation_workflow import trigger_workflow  # noqa: E402
from .logging_utils import get_logger, log_event  # noqa: E402
from .output_tool import (  # noqa: E402
    SUBMIT_RESOLUTION_SCHEMA,
    SUBMIT_RESOLUTION_TOOL_NAME,
    enforce_resolution,
    validate_schema,
)
from .prompts import SYSTEM_PROMPT  # noqa: E402
from .tool_loop import ModelAPIError, ToolCallRecord, ToolLoopResult, run_tool_loop  # noqa: E402

_logger = get_logger(__name__)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
# The SDK itself already retries connection errors and 408/409/429/5xx with
# exponential backoff (default 2). Only override it if the caller asked to.
_max_retries_env = os.environ.get("ANTHROPIC_MAX_RETRIES")
DEFAULT_MAX_RETRIES = int(_max_retries_env) if _max_retries_env else None


class ResolverAgent:
    """A single autonomous agent that resolves one GlobalCart support ticket
    per call to :meth:`resolve`.
    """

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 8,
        escalation_queue_path: Optional[Path] = None,
        escalation_writer: Optional[Callable[[Dict[str, Any], Path], None]] = None,
        require_verified_requester: bool = False,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations!r}.")
        if client is not None:
            self.client = client
        elif DEFAULT_MAX_RETRIES is not None:
            self.client = anthropic.Anthropic(max_retries=DEFAULT_MAX_RETRIES)
        else:
            self.client = anthropic.Anthropic()
        self.model = model
        self.max_iterations = max_iterations
        self.tool_schemas = list(gc.TOOL_SCHEMAS) + [SUBMIT_RESOLUTION_SCHEMA]
        self.tool_registry = dict(gc.TOOL_REGISTRY)
        # None means "use escalation_workflow.DEFAULT_QUEUE_PATH" -- an
        # explicit override here is mainly for tests and alternate deployments
        # that want the ops queue written somewhere other than the repo root.
        self.escalation_queue_path = escalation_queue_path
        # None means "use escalation_workflow's own default resolution"
        # (webhook if ESCALATION_WEBHOOK_URL is set, else the local file).
        # An explicit override here is for a caller that wants to wire in a
        # real ticketing system's SDK directly, bypassing the generic
        # webhook path entirely.
        self.escalation_writer = escalation_writer
        # A deployment posture, not a per-call choice: a customer-facing
        # deployment should never be able to silently fall back to
        # unrestricted mode just because a caller forgot to pass
        # requester_user_id on one call site. False (the default) preserves
        # today's behavior exactly -- appropriate for an internal
        # ops-console context where any record is fair game.
        self.require_verified_requester = require_verified_requester

    def resolve(self, ticket_text: str, requester_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Resolve one support ticket end to end.

        ``requester_user_id`` is the authenticated identity of whoever
        submitted this ticket -- this package has no authentication of its
        own (out of scope, an upstream concern: verifying that the caller
        actually *is* this identity, e.g. via a session token or SSO, has
        to happen upstream before this is called). If the caller supplies
        one, any tool result naming a *different* owning customer is denied
        (NOT_AUTHORIZED) before it ever reaches the model, rather than
        merely flagged afterward. Omitting it reproduces today's
        unrestricted behavior exactly -- e.g. an internal ops-console
        context where any record is fair game -- unless this agent was
        constructed with ``require_verified_requester=True``, in which case
        omitting it raises immediately instead of silently degrading to
        unrestricted (see below).

        Returns the submit_resolution arguments (reasoning_chain,
        action_taken, customer_response) plus bookkeeping fields:
        ``_case_id`` (a short id correlating this resolution's log lines),
        ``_tool_calls`` (a trace of every GlobalCart tool call made),
        ``_validation_warnings`` (any mismatch between the stated decision
        and what the tools actually returned), ``_corrections`` (what was
        deterministically overridden to fix a mismatch, if anything -- see
        output_tool.enforce_resolution), ``_stopped_reason`` (``"stop"``,
        ``"max_iterations"`` or ``"api_error"``), and ``_workflow_triggered``
        (whether an ops-queue record was written for ESCALATION_REQUIRED /
        CANNOT_RESOLVE -- see escalation_workflow.trigger_workflow).

        A resolution that is structurally invalid (missing required fields,
        or a ``decision`` outside the four allowed values -- see
        output_tool.validate_schema) is treated exactly like the model never
        calling ``submit_resolution`` at all: discarded in favor of the same
        safe fallback. A resolution that IS structurally valid but
        contradicts what the tools actually returned is not just flagged --
        ``enforce_resolution`` deterministically corrects the decision and
        customer_response to match the tools' ground truth before this
        method ever returns it.

        Only ``ModelAPIError`` -- a genuine API/infra failure, already
        distinguished by tool_loop.py from the SDK's own exhausted retries --
        is caught here and turned into a safe escalation. Any other
        exception is a real bug and is left to propagate, not swallowed.

        Raises:
            ValueError: if this agent was constructed with
                ``require_verified_requester=True`` and ``requester_user_id``
                was not supplied. A misconfigured customer-facing deployment
                should fail loudly and immediately, not silently process a
                ticket in unrestricted mode.
        """
        if self.require_verified_requester and requester_user_id is None:
            raise ValueError(
                "This ResolverAgent requires a verified requester_user_id "
                "(require_verified_requester=True) but resolve() was called "
                "without one -- refusing to run in unrestricted mode. Pass "
                "requester_user_id after verifying the caller's identity "
                "upstream, or construct with "
                "require_verified_requester=False for an internal "
                "ops-console context where any record is fair game."
            )

        case_id = uuid.uuid4().hex[:8]
        ctx = {"case_id": case_id}
        messages: List[Dict[str, Any]] = [{"role": "user", "content": ticket_text}]

        tool_registry = self.tool_registry
        if requester_user_id is not None:
            tool_registry = authorize_tool_registry(self.tool_registry, requester_user_id, ctx)

        try:
            result = run_tool_loop(
                client=self.client,
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=messages,
                tool_schemas=self.tool_schemas,
                tool_registry=tool_registry,
                stop_tool_name=SUBMIT_RESOLUTION_TOOL_NAME,
                max_iterations=self.max_iterations,
                log_context=ctx,
            )
        except ModelAPIError as exc:
            log_event(
                _logger,
                logging.ERROR,
                "agent.api_error",
                error_type=type(exc.original).__name__,
                tool_calls_so_far=len(exc.tool_calls),
                **ctx,
            )
            resolution = self._api_failure_resolution(exc)
            resolution["_case_id"] = case_id
            resolution["_tool_calls"] = [
                {"name": c.name, "input": c.input, "result": c.result} for c in exc.tool_calls
            ]
            resolution["_validation_warnings"] = []
            resolution["_corrections"] = []
            resolution["_stopped_reason"] = "api_error"
            return self._finalize_workflow(resolution, case_id, ctx)

        raw_resolution = self._extract_resolution(result.tool_calls)
        schema_errors = validate_schema(raw_resolution) if raw_resolution is not None else None
        used_fallback = raw_resolution is None or bool(schema_errors)

        warnings: List[str] = []
        corrections: List[str] = []
        if used_fallback:
            resolution = self._fallback_resolution(result, reason=schema_errors)
        else:
            resolution, warnings, corrections = enforce_resolution(raw_resolution, result.tool_calls)

        resolution["_case_id"] = case_id
        resolution["_tool_calls"] = [
            {"name": c.name, "input": c.input, "result": c.result}
            for c in result.tool_calls
            if c.name != SUBMIT_RESOLUTION_TOOL_NAME
        ]
        resolution["_validation_warnings"] = warnings
        resolution["_corrections"] = corrections
        resolution["_stopped_reason"] = result.stopped_reason

        if used_fallback:
            log_event(
                _logger,
                logging.WARNING,
                "agent.fallback_resolution_used",
                stopped_reason=result.stopped_reason,
                schema_errors=len(schema_errors) if schema_errors else 0,
                **ctx,
            )
        elif corrections:
            log_event(
                _logger,
                logging.WARNING,
                "agent.resolution_corrected",
                correction_count=len(corrections),
                decision=resolution.get("action_taken", {}).get("decision"),
                **ctx,
            )
        elif warnings:
            # Defensive fallback only -- given enforce_resolution corrects
            # every condition it detects, this branch should be unreachable
            # in practice, but stays as a safety net if that ever drifts.
            log_event(
                _logger,
                logging.WARNING,
                "agent.validation_warnings",
                warning_count=len(warnings),
                decision=resolution.get("action_taken", {}).get("decision"),
                **ctx,
            )
        else:
            log_event(
                _logger,
                logging.INFO,
                "agent.case_resolved",
                decision=resolution.get("action_taken", {}).get("decision"),
                tools_called=len(resolution["_tool_calls"]),
                stopped_reason=result.stopped_reason,
                **ctx,
            )

        return self._finalize_workflow(resolution, case_id, ctx)

    def _finalize_workflow(
        self, resolution: Dict[str, Any], case_id: str, ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage-9 "trigger workflow": write an ops-queue record for any case
        a human still has to act on (ESCALATION_REQUIRED, CANNOT_RESOLVE).

        Every path that can produce a final resolution -- the normal flow,
        the schema-fallback, and the API-failure fallback -- funnels through
        here, so a human always gets a real record to act on rather than
        only a customer-facing sentence claiming one exists. See
        escalation_workflow.trigger_workflow for the record shape and the
        (deliberately minimal) write mechanics.
        """
        record = trigger_workflow(
            resolution, case_id, queue_path=self.escalation_queue_path, writer=self.escalation_writer
        )
        resolution["_workflow_triggered"] = record is not None
        if record is not None:
            log_event(
                _logger,
                logging.INFO,
                "agent.workflow_triggered",
                decision=record["decision"],
                **ctx,
            )
        return resolution

    @staticmethod
    def _extract_resolution(tool_calls: List[ToolCallRecord]) -> Optional[Dict[str, Any]]:
        for call in reversed(tool_calls):
            if call.name == SUBMIT_RESOLUTION_TOOL_NAME:
                return dict(call.input)
        return None

    @staticmethod
    def _fallback_resolution(result: ToolLoopResult, reason: Optional[List[str]] = None) -> Dict[str, Any]:
        """A safe, structured escalation used for two distinct triggers:

        1. The model never called submit_resolution -- not even when forced
           to on the final turn (e.g. it produced plain text and stopped
           some other way). ``reason`` is None; the reasoning_chain
           explains this via ``stopped_reason``.
        2. The model DID call submit_resolution, but the result failed
           output_tool.validate_schema (missing required fields, or a
           decision outside the four allowed values). ``reason`` carries
           the schema errors, and a malformed call is treated exactly like
           no call at all -- there's no sensible way to patch individual
           fields on a structurally invalid resolution, so it's discarded
           in favor of this same safe fallback.

        Either way the agent must return something structured and safe
        instead of raising or passing through a broken shape, so it
        escalates rather than guessing.
        """
        if reason:
            explanation = f"The model's submit_resolution call was structurally invalid: {'; '.join(reason)}"
        else:
            explanation = (
                "The agent did not produce a structured resolution within "
                f"the allotted turns (stopped_reason={result.stopped_reason!r})."
            )
        return {
            "reasoning_chain": [explanation],
            "action_taken": {
                "tools_called": [c.name for c in result.tool_calls],
                "decision": "ESCALATION_REQUIRED",
                "refund_amount": None,
                "refund_id": None,
            },
            "customer_response": (
                "Thanks for reaching out. This case needs a closer look from "
                "our operations team before we can give you a final answer -- "
                "we're escalating it now and will follow up shortly."
            ),
        }

    @staticmethod
    def _api_failure_resolution(exc: ModelAPIError) -> Dict[str, Any]:
        """The Anthropic API call itself failed -- not a business decision
        the agent got wrong, but an infrastructure failure (already
        distinguished from the SDK's own retryable errors by tool_loop.py).
        The safe response is to escalate rather than guess, while keeping
        the failure visible in the audit trail instead of swallowing it.
        """
        return {
            "reasoning_chain": [f"The call to the model API failed: {exc}."],
            "action_taken": {
                "tools_called": [c.name for c in exc.tool_calls],
                "decision": "ESCALATION_REQUIRED",
                "refund_amount": None,
                "refund_id": None,
            },
            "customer_response": (
                "Sorry -- we're having a technical issue on our end and "
                "couldn't finish looking into this right now. I've flagged "
                "your case for a member of our team to follow up shortly."
            ),
        }
