"""Generic Anthropic tool-calling loop.

Deliberately knows nothing about GlobalCart, refunds, or any specific domain --
it only knows how to run the mechanical tool_use / tool_result cycle described
in Anthropic's tool-use docs (https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview).
This is the substrate Part 2's multi-agent system can reuse unchanged for
every agent in the team, swapping in only different prompts and tool sets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import anthropic

from .logging_utils import get_logger, log_event

_logger = get_logger(__name__)


@dataclass
class ToolCallRecord:
    name: str
    input: Dict[str, Any]
    result: Any


@dataclass
class ToolLoopResult:
    final_message: Any
    messages: List[Dict[str, Any]]
    tool_calls: List[ToolCallRecord]
    stopped_reason: str  # "stop" | "max_iterations"


class ToolExecutionError(RuntimeError):
    """A genuine programmer error surfaced by a tool (e.g. wrong argument type).

    Business failures are not exceptions -- mock_services.py already returns
    them as ``{"error": ...}`` dicts, which flow back to the model as an
    ordinary tool_result. This exists only so a real bug in a tool call still
    surfaces loudly instead of being swallowed.
    """


class ModelAPIError(RuntimeError):
    """The call to the Anthropic API itself failed.

    The SDK already retries connection errors and 408/409/429/5xx responses
    internally (``max_retries``, default 2, with exponential backoff) --
    this is raised only for what reaches us afterward: retries exhausted, or
    an immediately non-retryable error (e.g. 400/401/403/404, such as the
    "credit balance too low" 400 hit during live testing). Carries whatever
    tool calls had already completed before the failure, so a caller doesn't
    lose the partial audit trail.
    """

    def __init__(self, tool_calls: List["ToolCallRecord"], original: BaseException) -> None:
        super().__init__(f"{type(original).__name__}: {original}")
        self.tool_calls = tool_calls
        self.original = original


def _signature(name: str, tool_input: Dict[str, Any]) -> tuple:
    return (name, tuple(sorted(tool_input.items())))


def _stringify(result: Any) -> str:
    try:
        return json.dumps(result)
    except TypeError:
        return str(result)


def _create(client: "anthropic.Anthropic", tool_calls_so_far: List[ToolCallRecord], **kwargs: Any) -> Any:
    try:
        return client.messages.create(**kwargs)
    except anthropic.APIError as exc:
        raise ModelAPIError(list(tool_calls_so_far), exc) from exc


def run_tool_loop(
    client: "anthropic.Anthropic",
    model: str,
    system: str,
    messages: List[Dict[str, Any]],
    tool_schemas: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable[..., Any]],
    *,
    stop_tool_name: Optional[str] = None,
    max_iterations: int = 8,
    temperature: Optional[float] = None,
    max_tokens: int = 2048,
    log_context: Optional[Dict[str, Any]] = None,
) -> ToolLoopResult:
    """Run send -> tool_use -> tool_result -> send until the model stops.

    ``messages`` is mutated in place (Anthropic's own convention) and also
    returned, so the caller keeps the full transcript for logging/validation.

    Stops when:
      * the model's turn calls ``stop_tool_name`` (the caller's "I am
        finished" signal -- used for ``submit_resolution`` here), or
      * the model's ``stop_reason`` is not ``tool_use`` (it produced plain
        text instead of calling a tool), or
      * ``max_iterations`` rounds pass without either -- the loop then forces
        one last turn with ``tool_choice`` pinned to ``stop_tool_name``, so
        the agent always terminates with a valid structured answer instead of
        spinning forever.

    A tool call repeated with the exact same arguments is not re-executed --
    a synthetic tool_result tells the model the retry was refused. This is
    what stops a confused agent from looping on the same failing call.

    Raises:
        ValueError: if ``max_iterations`` is not at least 1. The loop's
            whole contract is "make at least one round trip to the model" --
            with ``max_iterations <= 0`` and no ``stop_tool_name`` set, the
            for-loop body would never run and the forced-final-call branch
            (gated on ``stop_tool_name``) wouldn't either, silently handing
            the caller a ``ToolLoopResult`` with ``final_message=None``
            instead of failing.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations!r}.")

    ctx = log_context or {}
    seen_calls: set = set()
    tool_calls: List[ToolCallRecord] = []
    response = None
    # Some models (e.g. claude-sonnet-5) reject an explicit `temperature` --
    # only pass it through when the caller actually asked for one.
    extra_kwargs = {} if temperature is None else {"temperature": temperature}

    for _ in range(max_iterations):
        response = _create(
            client,
            tool_calls,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tool_schemas,
            **extra_kwargs,
        )

        if response.stop_reason != "tool_use":
            return ToolLoopResult(response, messages, tool_calls, "stop")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        called_stop_tool = False
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == stop_tool_name:
                called_stop_tool = True
                tool_calls.append(ToolCallRecord(block.name, block.input, None))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Resolution recorded.",
                    }
                )
                continue

            sig = _signature(block.name, block.input)
            if sig in seen_calls:
                log_event(_logger, logging.WARNING, "tool_loop.repeat_call_refused", tool=block.name, **ctx)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            "You already called this tool with these exact "
                            "arguments -- the result will not change. Do not "
                            "retry it. Use the result you already have, or "
                            "escalate if it is not enough to decide."
                        ),
                    }
                )
                continue
            seen_calls.add(sig)

            fn = tool_registry.get(block.name)
            if fn is None:
                log_event(_logger, logging.WARNING, "tool_loop.unknown_tool_requested", tool=block.name, **ctx)
                result: Any = {
                    "error": "UNKNOWN_TOOL",
                    "message": f"No such tool: {block.name}",
                }
            else:
                try:
                    result = fn(**block.input)
                except TypeError as exc:
                    raise ToolExecutionError(
                        f"Programmer error calling {block.name}({block.input!r}): {exc}"
                    ) from exc
                log_event(
                    _logger,
                    logging.DEBUG,
                    "tool_loop.tool_executed",
                    tool=block.name,
                    is_error=isinstance(result, dict) and "error" in result,
                    **ctx,
                )

            tool_calls.append(ToolCallRecord(block.name, block.input, result))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _stringify(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

        if called_stop_tool:
            return ToolLoopResult(response, messages, tool_calls, "stop")

    if stop_tool_name is not None:
        log_event(_logger, logging.WARNING, "tool_loop.max_iterations_reached", max_iterations=max_iterations, **ctx)
        response = _create(
            client,
            tool_calls,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tool_schemas,
            tool_choice={"type": "tool", "name": stop_tool_name},
            **extra_kwargs,
        )
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "tool_use" and block.name == stop_tool_name:
                tool_calls.append(ToolCallRecord(block.name, block.input, None))

    return ToolLoopResult(response, messages, tool_calls, "max_iterations")
