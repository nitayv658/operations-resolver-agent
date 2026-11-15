"""Tests for resolver_agent_crewai.provider -- the one place the CrewAI port
decides which LLM provider a bare model name belongs to and which exception
types mean "the model API failed" (as opposed to a bug in this code).

None of these need a live model call or a provider key.
"""

from __future__ import annotations

import httpx
import openai
import pydantic
import pytest

from resolver_agent_crewai.provider import (
    default_comms_model,
    is_model_api_error,
    qualify_model,
)

from ..helpers import fake_bad_request_error


def _openai_connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1/chat"))


def test_qualify_model_when_unprefixed_should_default_to_the_anthropic_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert qualify_model("claude-sonnet-5") == "anthropic/claude-sonnet-5"


def test_qualify_model_when_already_prefixed_should_be_left_alone(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    assert qualify_model("openai/gpt-4o") == "openai/gpt-4o"


def test_qualify_model_when_llm_provider_env_is_set_should_use_it_as_the_prefix(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    assert qualify_model("gemini-2.5-flash") == "gemini/gemini-2.5-flash"


def test_qualify_model_when_llm_provider_env_is_blank_should_fall_back_to_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "   ")

    assert qualify_model("claude-sonnet-5") == "anthropic/claude-sonnet-5"


def test_default_comms_model_when_shared_model_is_anthropic_should_use_the_cheaper_haiku_default():
    assert default_comms_model("anthropic/claude-sonnet-5") == "anthropic/claude-haiku-4-5-20251001"


def test_default_comms_model_when_shared_model_is_not_anthropic_should_follow_it_instead():
    # The Haiku default only exists on Anthropic -- a gemini/ or openai/ run
    # with it would silently demand a second provider's API key.
    assert default_comms_model("openai/gpt-4o") == "openai/gpt-4o"


def test_is_model_api_error_when_anthropic_error_should_be_true():
    assert is_model_api_error(fake_bad_request_error("boom")) is True


def test_is_model_api_error_when_openai_error_should_be_true():
    # Also the base class of LiteLLM's exceptions, so this covers both the
    # native OpenAI provider and CrewAI's LiteLLM fallback.
    assert is_model_api_error(_openai_connection_error()) is True


def test_is_model_api_error_when_google_genai_error_should_be_true_if_installed():
    errors = pytest.importorskip("google.genai.errors")

    assert is_model_api_error(errors.APIError(500, {"error": {"message": "boom"}})) is True


@pytest.mark.parametrize("exc", [ValueError("x"), KeyError("x"), RuntimeError("x")])
def test_is_model_api_error_when_ordinary_bug_should_be_false(exc):
    # A bug in this code must still surface, not be disguised as an outage.
    assert is_model_api_error(exc) is False


def test_is_model_api_error_when_pydantic_validation_error_should_be_false():
    # Output-parse failures are a different failure mode with their own
    # handling in the orchestrator -- not an API outage.
    class _M(pydantic.BaseModel):
        x: int

    with pytest.raises(pydantic.ValidationError) as caught:
        _M(x="not an int")

    assert is_model_api_error(caught.value) is False
