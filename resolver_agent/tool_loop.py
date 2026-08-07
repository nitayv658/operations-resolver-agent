"""Generic Anthropic tool-calling loop.

Deliberately knows nothing about GlobalCart, refunds, or any specific domain --
it only knows how to run the mechanical tool_use / tool_result cycle described
in Anthropic's tool-use docs (https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview).
This is the substrate Part 2's multi-agent system can reuse unchanged for
every agent in the team, swapping in only different prompts and tool sets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import anthropic


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


def _signature(name: str, tool_input: Dict[str, Any]) -> tuple:
    return (name, tuple(sorted(tool_input.items())))


def _stringify(result: Any) -> str:
    try:
        return json.dumps(result)
    except TypeError:
        return str(result)


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
    """
    seen_calls: set = set()
    tool_calls: List[ToolCallRecord] = []
    response = None
    # Some models (e.g. claude-sonnet-5) reject an explicit `temperature` --
    # only pass it through when the caller actually asked for one.
    extra_kwargs = {} if temperature is None else {"temperature": temperature}

    for _ in range(max_iterations):
        response = client.messages.create(
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
        response = client.messages.create(
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
