"""Test doubles for the Anthropic client.

These stand in for the *model's decisions* only -- no network call, no API
key needed. Everything downstream of a scripted response (tool dispatch via
the real starter-kit mock_services, the loop's own dedup/guard logic, the
output validator) runs as real, unmocked code. This is deliberate: mocking
mock_services itself would just be testing that a mock was called, not that
the agent's logic works (see Anti-Pattern 1/3 in the review protocol).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

_counter = [0]


def tool_use_block(name: str, input_: Dict[str, Any]) -> SimpleNamespace:
    """A stand-in for anthropic's ToolUseBlock -- exposes exactly the
    attributes tool_loop.py reads: .type, .id, .name, .input."""
    _counter[0] += 1
    return SimpleNamespace(type="tool_use", id=f"toolu_{_counter[0]}", name=name, input=input_)


def text_block(text: str) -> SimpleNamespace:
    """A stand-in for anthropic's TextBlock -- tool_loop.py only reads .type
    on non-tool_use blocks, but .text is included for realism."""
    return SimpleNamespace(type="text", text=text)


class ScriptedResponse:
    """A stand-in for anthropic's Message -- exposes .content and .stop_reason."""

    def __init__(self, blocks: List[SimpleNamespace], stop_reason: str = "tool_use"):
        self.content = blocks
        self.stop_reason = stop_reason


class ScriptedClient:
    """A stand-in for anthropic.Anthropic(). Returns one ScriptedResponse per
    call to .messages.create(), in order, regardless of what arguments (model,
    system, messages, tools, tool_choice, ...) it was called with."""

    def __init__(self, script: List[ScriptedResponse]):
        self._script = list(script)
        self.calls = 0

    class _Messages:
        def __init__(self, outer: "ScriptedClient"):
            self._outer = outer

        def create(self, **kwargs: Any) -> ScriptedResponse:
            self._outer.calls += 1
            if not self._outer._script:
                raise AssertionError(
                    "ScriptedClient ran out of scripted responses -- the loop "
                    "made more calls to messages.create() than the test expected."
                )
            return self._outer._script.pop(0)

    @property
    def messages(self) -> "ScriptedClient._Messages":
        return self._Messages(self)
