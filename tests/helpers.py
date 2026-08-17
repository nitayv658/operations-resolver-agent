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
from typing import Any, Dict, List, Union

import anthropic
import httpx

_counter = [0]


def fake_api_connection_error(message: str = "Connection error.") -> anthropic.APIConnectionError:
    """A real anthropic.APIConnectionError -- the base class our code catches
    for retryable-but-exhausted / connection-level failures."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message=message, request=request)


def fake_bad_request_error(message: str = "Your credit balance is too low.") -> anthropic.BadRequestError:
    """A real anthropic.BadRequestError (400) -- the exact class hit live
    when the account ran out of credits. Not retried by the SDK itself."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return anthropic.BadRequestError(message, response=response, body={"error": {"message": message}})


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
    system, messages, tools, tool_choice, ...) it was called with. A script
    entry that is an exception instance is raised instead of returned --
    simulating the SDK surfacing an API error after its own internal retries
    (if any) are exhausted."""

    def __init__(self, script: List[Union[ScriptedResponse, BaseException]]):
        self._script: List[Union[ScriptedResponse, BaseException]] = list(script)
        self.calls = 0
        self.call_kwargs: List[Dict[str, Any]] = []

    class _Messages:
        def __init__(self, outer: "ScriptedClient"):
            self._outer = outer

        def create(self, **kwargs: Any) -> ScriptedResponse:
            self._outer.calls += 1
            self._outer.call_kwargs.append(kwargs)
            if not self._outer._script:
                raise AssertionError(
                    "ScriptedClient ran out of scripted responses -- the loop "
                    "made more calls to messages.create() than the test expected."
                )
            item = self._outer._script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    @property
    def messages(self) -> "ScriptedClient._Messages":
        return self._Messages(self)
