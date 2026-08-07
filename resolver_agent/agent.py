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
from .tool_loop import ToolCallRecord, ToolLoopResult, run_tool_loop  # noqa: E402

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


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
        self.client = client or anthropic.Anthropic()
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
        and what the tools actually returned) and ``_stopped_reason``.
        """
        messages: List[Dict[str, Any]] = [{"role": "user", "content": ticket_text}]

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
