"""Provider-neutral model plumbing for the CrewAI port.

The hand-rolled version (resolver_agent/) speaks Anthropic's tool-use
protocol directly and cannot change provider without an adapter. The CrewAI
port can -- CrewAI routes any ``provider/model`` string -- but it used to
hard-code two Anthropic assumptions, both centralised here instead:

  * Which provider a bare model name belongs to: ``qualify_model()``
    prefixes it with ``LLM_PROVIDER`` (default ``anthropic``, so existing
    behavior is unchanged).
  * Which exceptions mean "the model API failed": ``MODEL_API_ERRORS``. Each
    provider's SDK has its own hierarchy -- ``anthropic.APIError``,
    ``openai.APIError`` (also the base class of LiteLLM's exceptions, so it
    covers CrewAI's LiteLLM fallback too) and ``google.genai.errors.APIError``
    -- so the tuple holds whichever of those are importable. Ordinary bugs
    (``ValueError``, ``KeyError``, ...) are deliberately *not* in it: they
    must still surface instead of being disguised as an outage.

Using a non-Anthropic provider also needs that provider's CrewAI extra (e.g.
``crewai[google-genai]``) and API key -- see requirements-crewai.txt. Weaker
models are likelier to trip the structured-output parsing the port already
works around (see the README's Part 2b), so a provider swap is a
reliability experiment, not just a config change.
"""

from __future__ import annotations

import importlib
import os
from typing import Tuple, Type

from resolver_agent.crew.orchestrator import DEFAULT_COMMS_MODEL

_DEFAULT_PROVIDER = "anthropic"

# (module, attribute) pairs, tried in order; a missing optional SDK is skipped.
_API_ERROR_CANDIDATES = (
    ("anthropic", "APIError"),
    ("openai", "APIError"),
    ("google.genai.errors", "APIError"),
)


def _load_api_error_types() -> Tuple[Type[BaseException], ...]:
    found = []
    for module_name, attr in _API_ERROR_CANDIDATES:
        try:
            found.append(getattr(importlib.import_module(module_name), attr))
        except (ImportError, AttributeError):
            continue
    return tuple(found)


MODEL_API_ERRORS: Tuple[Type[BaseException], ...] = _load_api_error_types()


def is_model_api_error(exc: BaseException) -> bool:
    """True if ``exc`` is a model-provider API failure (any supported SDK)."""
    return isinstance(exc, MODEL_API_ERRORS)


def _provider() -> str:
    return (os.environ.get("LLM_PROVIDER") or "").strip() or _DEFAULT_PROVIDER


def qualify_model(model: str) -> str:
    """CrewAI needs a provider-prefixed model string; the shared
    ANTHROPIC_MODEL-style names are bare. Already-prefixed names pass through
    untouched, so ``openai/gpt-4o`` works regardless of ``LLM_PROVIDER``.
    """
    return model if "/" in model else f"{_provider()}/{model}"


def default_comms_model(shared_model: str) -> str:
    """Comms's default when nothing overrides it.

    On Anthropic it is the cheaper Haiku default the hand-rolled crew uses
    (Comms is low-stakes, regex-checked text). Haiku only exists on
    Anthropic, so for any other provider it follows the shared model --
    otherwise a gemini/ or openai/ run would silently demand a second
    provider's API key.
    """
    if shared_model.startswith(f"{_DEFAULT_PROVIDER}/"):
        return f"{_DEFAULT_PROVIDER}/{DEFAULT_COMMS_MODEL}"
    return shared_model
