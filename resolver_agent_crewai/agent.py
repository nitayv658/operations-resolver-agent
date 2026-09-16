"""CrewResolverAgent -- Part 1's resolver, rebuilt on CrewAI instead of the
hand-rolled tool_loop.

Same public contract as resolver_agent.agent.ResolverAgent: construct once,
call .resolve(ticket_text, requester_user_id=None) per ticket, get back the
same submit_resolution-shaped dict plus the same bookkeeping fields
(_case_id, _tool_calls, _validation_warnings, _corrections, _stopped_reason,
_workflow_triggered). Reuses everything from resolver_agent/ that isn't
loop-specific: output_tool's validate_schema/enforce_resolution,
escalation_workflow.trigger_workflow, logging_utils, and (via tools.py)
authorization.authorize_tool_registry.

Real mechanism-level differences from Part 1, kept rather than papered over
(see the root README's CrewAI-port section):
  * Structured output comes from CrewAI's Task.output_pydantic parsing the
    agent's final answer, not a forced tool_use call pinned to a stop tool
    (tool_loop._forced_stop_call has no equivalent here).
  * With the `anthropic/` model prefix _litellm_model() always sets, CrewAI
    routes calls through its native Anthropic provider (crewai[anthropic]
    extra -- requirements-crewai.txt), which calls the real anthropic SDK
    directly and re-raises its own exceptions largely unchanged. So
    anthropic.APIError -- the exact type tool_loop.py already catches --
    is what's caught below too, confirmed empirically (not guessed): a
    forced auth failure through this same code path raised
    anthropic.AuthenticationError, a real subclass of anthropic.APIError.
    Still coarser than tool_loop.ModelAPIError: it can't distinguish "SDK
    retries exhausted" from "immediately non-retryable" the way that type
    does, and it carries no equivalent of ModelAPIError.tool_calls (the
    call_log list below already serves that purpose here).
  * No equivalent of tool_loop's Anthropic prompt-caching (cache_control
    breakpoints) is wired up.
  * stopped_reason collapses to "stop" (or "api_error") -- CrewAI does not
    surface a clean "iteration budget exhausted" signal the way
    tool_loop.ToolLoopResult.stopped_reason does.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

import anthropic  # noqa: E402  (crewai's native Anthropic provider re-raises this)
from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402

from resolver_agent.escalation_workflow import trigger_workflow  # noqa: E402
from resolver_agent.logging_utils import get_logger, log_event  # noqa: E402
from resolver_agent.output_tool import enforce_resolution, validate_schema  # noqa: E402
from resolver_agent.tool_loop import ToolCallRecord  # noqa: E402

from .prompts import BACKSTORY, GOAL, ROLE, TASK_DESCRIPTION, TASK_EXPECTED_OUTPUT  # noqa: E402
from .schemas import Resolution  # noqa: E402
from .tools import build_tools  # noqa: E402

# Nested under the "resolver_agent" logger tree on purpose, not
# "resolver_agent_crewai.agent" (__name__) -- logging_utils.configure_logging()
# attaches its stderr JSON handler and sets propagate=False on exactly (and
# only) the "resolver_agent" logger, so a sibling top-level namespace would
# silently never reach that handler at all.
_logger = get_logger("resolver_agent.crewai")

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def _litellm_model(model: str) -> str:
    """CrewAI calls LiteLLM under the hood, which needs a provider-prefixed
    model string -- Part 1's ANTHROPIC_MODEL / default is unprefixed."""
    return model if "/" in model else f"anthropic/{model}"


class CrewResolverAgent:
    """A single autonomous CrewAI agent that resolves one GlobalCart support
    ticket per call to :meth:`resolve` -- the CrewAI-framework counterpart
    to resolver_agent.agent.ResolverAgent.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 8,
        escalation_queue_path: Optional[Path] = None,
        escalation_writer: Optional[Callable[[Dict[str, Any], Path], None]] = None,
        require_verified_requester: bool = False,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations!r}.")
        self.model = _litellm_model(model)
        # max_tokens=4096 mirrors tool_loop.py's own default -- raised there
        # after a live Part 2 run hit the API's 1024 default mid-turn with no
        # tool call at all (see tool_loop.py's run_tool_loop docstring).
        self.llm = LLM(model=self.model, max_tokens=4096)
        self.max_iterations = max_iterations
        self.escalation_queue_path = escalation_queue_path
        self.escalation_writer = escalation_writer
        self.require_verified_requester = require_verified_requester

    def resolve(self, ticket_text: str, requester_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Resolve one support ticket end to end -- see
        resolver_agent.agent.ResolverAgent.resolve() for the full contract
        this mirrors (requester_user_id semantics, fallback/correction
        behavior, bookkeeping fields).
        """
        if self.require_verified_requester and requester_user_id is None:
            raise ValueError(
                "This CrewResolverAgent requires a verified requester_user_id "
                "(require_verified_requester=True) but resolve() was called "
                "without one -- refusing to run in unrestricted mode. Pass "
                "requester_user_id after verifying the caller's identity "
                "upstream, or construct with require_verified_requester=False "
                "for an internal ops-console context where any record is "
                "fair game."
            )

        case_id = uuid.uuid4().hex[:8]
        ctx = {"case_id": case_id}
        call_log: List[ToolCallRecord] = []
        tools = build_tools(call_log, requester_user_id, ctx)

        agent = Agent(
            role=ROLE,
            goal=GOAL,
            backstory=BACKSTORY,
            tools=tools,
            llm=self.llm,
            max_iter=self.max_iterations,
            verbose=False,
        )
        task = Task(
            description=TASK_DESCRIPTION.format(ticket_text=ticket_text),
            expected_output=TASK_EXPECTED_OUTPUT,
            agent=agent,
            output_pydantic=Resolution,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        try:
            crew.kickoff()
        except anthropic.APIError as exc:  # see module docstring: confirmed empirically
            log_event(
                _logger,
                logging.ERROR,
                "crew_agent.api_error",
                error_type=type(exc).__name__,
                tool_calls_so_far=len(call_log),
                **ctx,
            )
            resolution = self._api_failure_resolution(exc, call_log)
            resolution["_case_id"] = case_id
            resolution["_tool_calls"] = [
                {"name": c.name, "input": c.input, "result": c.result} for c in call_log
            ]
            resolution["_validation_warnings"] = []
            resolution["_corrections"] = []
            resolution["_stopped_reason"] = "api_error"
            return self._finalize_workflow(resolution, case_id, ctx)

        raw_resolution = self._extract_resolution(task)
        schema_errors = validate_schema(raw_resolution) if raw_resolution is not None else None
        used_fallback = raw_resolution is None or bool(schema_errors)

        warnings: List[str] = []
        corrections: List[str] = []
        if used_fallback:
            resolution = self._fallback_resolution(call_log, reason=schema_errors)
        else:
            resolution, warnings, corrections = enforce_resolution(raw_resolution, call_log)

        resolution["_case_id"] = case_id
        resolution["_tool_calls"] = [{"name": c.name, "input": c.input, "result": c.result} for c in call_log]
        resolution["_validation_warnings"] = warnings
        resolution["_corrections"] = corrections
        resolution["_stopped_reason"] = "stop"

        if used_fallback:
            log_event(
                _logger,
                logging.WARNING,
                "crew_agent.fallback_resolution_used",
                schema_errors=len(schema_errors) if schema_errors else 0,
                **ctx,
            )
        elif corrections:
            log_event(
                _logger,
                logging.WARNING,
                "crew_agent.resolution_corrected",
                correction_count=len(corrections),
                decision=resolution.get("action_taken", {}).get("decision"),
                **ctx,
            )
        else:
            log_event(
                _logger,
                logging.INFO,
                "crew_agent.case_resolved",
                decision=resolution.get("action_taken", {}).get("decision"),
                tools_called=len(resolution["_tool_calls"]),
                **ctx,
            )

        return self._finalize_workflow(resolution, case_id, ctx)

    def _finalize_workflow(self, resolution: Dict[str, Any], case_id: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        record = trigger_workflow(
            resolution, case_id, queue_path=self.escalation_queue_path, writer=self.escalation_writer
        )
        resolution["_workflow_triggered"] = record is not None
        if record is not None:
            log_event(_logger, logging.INFO, "crew_agent.workflow_triggered", decision=record["decision"], **ctx)
        return resolution

    @staticmethod
    def _extract_resolution(task: Task) -> Optional[Dict[str, Any]]:
        output = task.output
        if output is None or output.pydantic is None:
            return None
        return output.pydantic.model_dump()

    @staticmethod
    def _fallback_resolution(call_log: List[ToolCallRecord], reason: Optional[List[str]] = None) -> Dict[str, Any]:
        if reason:
            explanation = f"The agent's final answer was structurally invalid: {'; '.join(reason)}"
        else:
            explanation = "The agent did not produce a structured resolution matching the required schema."
        return {
            "reasoning_chain": [explanation],
            "action_taken": {
                "tools_called": [c.name for c in call_log],
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
    def _api_failure_resolution(exc: "anthropic.APIError", call_log: List[ToolCallRecord]) -> Dict[str, Any]:
        return {
            "reasoning_chain": [f"The call to the model API failed: {exc}."],
            "action_taken": {
                "tools_called": [c.name for c in call_log],
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
