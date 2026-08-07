"""ResolverAgent -- wires the GlobalCart tools and the submit_resolution
output contract into the generic tool_loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from .output_tool import (  # noqa: E402
    SUBMIT_RESOLUTION_SCHEMA,
    SUBMIT_RESOLUTION_TOOL_NAME,
    validate_resolution,
)
from .prompts import SYSTEM_PROMPT  # noqa: E402
from .tool_loop import ModelAPIError, ToolCallRecord, ToolLoopResult, run_tool_loop  # noqa: E402

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
    ) -> None:
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

    def resolve(self, ticket_text: str) -> Dict[str, Any]:
        """Resolve one support ticket end to end.

        Returns the submit_resolution arguments (reasoning_chain,
        action_taken, customer_response) plus bookkeeping fields:
        ``_tool_calls`` (a trace of every GlobalCart tool call made),
        ``_validation_warnings`` (any mismatch between the stated decision
        and what the tools actually returned) and ``_stopped_reason``
        (``"stop"``, ``"max_iterations"`` or ``"api_error"``).

        Only ``ModelAPIError`` -- a genuine API/infra failure, already
        distinguished by tool_loop.py from the SDK's own exhausted retries --
        is caught here and turned into a safe escalation. Any other
        exception is a real bug and is left to propagate, not swallowed.
        """
        messages: List[Dict[str, Any]] = [{"role": "user", "content": ticket_text}]

        try:
            result = run_tool_loop(
                client=self.client,
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=messages,
                tool_schemas=self.tool_schemas,
                tool_registry=self.tool_registry,
                stop_tool_name=SUBMIT_RESOLUTION_TOOL_NAME,
                max_iterations=self.max_iterations,
            )
        except ModelAPIError as exc:
            resolution = self._api_failure_resolution(exc)
            resolution["_tool_calls"] = [
                {"name": c.name, "input": c.input, "result": c.result} for c in exc.tool_calls
            ]
            resolution["_validation_warnings"] = []
            resolution["_stopped_reason"] = "api_error"
            return resolution

        resolution = self._extract_resolution(result.tool_calls)
        if resolution is None:
            resolution = self._fallback_resolution(result)

        resolution["_tool_calls"] = [
            {"name": c.name, "input": c.input, "result": c.result}
            for c in result.tool_calls
            if c.name != SUBMIT_RESOLUTION_TOOL_NAME
        ]
        resolution["_validation_warnings"] = validate_resolution(resolution, result.tool_calls)
        resolution["_stopped_reason"] = result.stopped_reason
        return resolution

    @staticmethod
    def _extract_resolution(tool_calls: List[ToolCallRecord]) -> Optional[Dict[str, Any]]:
        for call in reversed(tool_calls):
            if call.name == SUBMIT_RESOLUTION_TOOL_NAME:
                return dict(call.input)
        return None

    @staticmethod
    def _fallback_resolution(result: ToolLoopResult) -> Dict[str, Any]:
        """The model never called submit_resolution -- not even when forced
        to on the final turn (e.g. it produced plain text and stopped some
        other way). The agent must still return something structured and
        safe instead of raising, so it escalates rather than guessing.
        """
        return {
            "reasoning_chain": [
                "The agent did not produce a structured resolution within "
                f"the allotted turns (stopped_reason={result.stopped_reason!r})."
            ],
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
