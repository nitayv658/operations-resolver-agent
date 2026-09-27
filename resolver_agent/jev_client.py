"""A small, swappable interface to "Jev" (TypeSafe AI's classification API).

Jev is a paid third-party cloud API, not a framework -- it offers fast
Choice/Score/Noul primitives for cheap, calibrated classification decisions,
meant to sit in front of an expensive LLM call. There is no verified real
``typesafe-sdk`` package to import (the only source is a marketing article);
this module exists so the rest of the codebase can depend on a stable
interface (:class:`JevClient`) rather than that speculative package directly.
:class:`MockJevClient` is the default and needs no credentials -- the live
path (:class:`JevAPIClient`) is a best-effort guess at the real shape, kept
behind a lazy import so its absence never breaks importing this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol


@dataclass(frozen=True)
class JevScore:
    """One Jev ``Score`` result, normalized to this codebase's own 0.0-1.0
    scale regardless of whatever raw scale a real provider might use.

    ``raw`` carries the untouched provider payload (or, for
    :class:`MockJevClient`, a small dict naming which heuristic fired) --
    kept only for logging/debugging, never read by decision logic.
    """

    score: float
    confidence: float
    raw: Any = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"JevScore.score must be in [0.0, 1.0], got {self.score!r}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"JevScore.confidence must be in [0.0, 1.0], got {self.confidence!r}.")


class JevClient(Protocol):
    """Anything that can score a structured query. Both
    :class:`MockJevClient` and :class:`JevAPIClient` satisfy this without
    subclassing it -- callers should type-hint against this, not either
    concrete class."""

    def score(self, query: Dict[str, Any]) -> JevScore: ...


def _default_scorer(query: Dict[str, Any]) -> JevScore:
    """A deterministic, no-credentials-needed stand-in for a real Jev call.

    Derives a plausible score/confidence straight from the query's own
    grounded fields (``eligible``, ``requires_escalation``,
    ``blocks_automatic_refund``) rather than anything resembling real
    classification -- good enough to exercise the gate's branching logic in
    tests and local runs, not a substitute for real calibration. Tests that
    need a specific branch should inject their own ``scorer`` instead of
    relying on this heuristic.
    """
    policy = query.get("check_return_policy") or {}
    risk = query.get("risk_report") or {}

    if policy.get("eligible") is False:
        return JevScore(score=0.05, confidence=0.9, raw={"heuristic": "ineligible"})

    if (
        policy.get("eligible") is True
        and not policy.get("requires_escalation")
        and not risk.get("blocks_automatic_refund")
    ):
        return JevScore(score=0.95, confidence=0.9, raw={"heuristic": "clean_eligible"})

    return JevScore(score=0.5, confidence=0.4, raw={"heuristic": "ambiguous"})


class MockJevClient:
    """Deterministic stand-in usable with no credentials or network access.

    ``scorer`` lets a test force a specific score/confidence regardless of
    the query's contents -- pass a callable instead of relying on
    :func:`_default_scorer`'s heuristic when a test needs a guaranteed
    branch.
    """

    def __init__(self, scorer: Optional[Callable[[Dict[str, Any]], JevScore]] = None) -> None:
        self._scorer = scorer or _default_scorer

    def score(self, query: Dict[str, Any]) -> JevScore:
        return self._scorer(query)


class JevAPIClient:
    """The real client -- speculative, since no verified ``typesafe-sdk``
    package exists beyond the dev.to article this was designed from.

    Imports ``typesafe_sdk`` lazily, inside ``__init__``, not at module
    import time -- so ``import jev_client`` never fails just because the
    package isn't installed (the default ``JEV_MODE=mock`` never reaches
    this class at all). If a real SDK's actual shape differs, only this
    class needs to change -- callers depend on :class:`JevClient`, not this.
    """

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 1.0) -> None:
        resolved_key = api_key or os.environ.get("JEV_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "JevAPIClient requires an API key -- pass api_key= or set "
                "JEV_API_KEY. (Set JEV_MODE=mock, the default, to avoid "
                "needing a real Jev account entirely.)"
            )
        try:
            import typesafe_sdk  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "JevAPIClient needs the 'typesafe-sdk' package, which is not "
                "installed. Install it via requirements-jev.txt, or set "
                "JEV_MODE=mock (the default) to use MockJevClient instead. "
                "Note: this package's real existence/shape is unverified -- "
                "see this module's docstring."
            ) from exc
        self._client = typesafe_sdk.Client(api_key=resolved_key, timeout=timeout_s)

    def score(self, query: Dict[str, Any]) -> JevScore:
        response = self._client.score(
            instructions=(
                "Estimate the probability that this GlobalCart refund claim "
                "is legitimate and should be auto-approved, given the "
                "grounded policy/fraud facts already looked up."
            ),
            criteria=query,
        )
        # Speculative normalization -- the article describes a 1-10-ish
        # scale; this assumes the SDK exposes 0-1 floats directly or that
        # normalizing is the caller's job. Unverified against a real SDK.
        return JevScore(score=float(response.score), confidence=float(response.confidence), raw=response)


def get_jev_client() -> JevClient:
    """Reads ``JEV_MODE`` (default ``"mock"``) and constructs the matching
    client. ``"mock"`` needs no other configuration; ``"live"`` reads
    ``JEV_API_KEY``."""
    mode = os.environ.get("JEV_MODE", "mock").lower()
    if mode == "mock":
        return MockJevClient()
    if mode == "live":
        return JevAPIClient()
    raise ValueError(f"JEV_MODE must be 'mock' or 'live', got {mode!r}.")
