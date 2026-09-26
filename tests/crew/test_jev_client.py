"""Tests for resolver_agent.jev_client -- the Jev SDK wrapper. No real
network access or credentials are ever needed: MockJevClient is the default,
and JevAPIClient is only exercised for its own error paths (missing
package/API key), never for a real call."""

from __future__ import annotations

import pytest

from resolver_agent.jev_client import JevAPIClient, JevScore, MockJevClient, get_jev_client


def test_jev_score_rejects_an_out_of_range_score():
    with pytest.raises(ValueError):
        JevScore(score=1.5, confidence=0.5)


def test_jev_score_rejects_an_out_of_range_confidence():
    with pytest.raises(ValueError):
        JevScore(score=0.5, confidence=-0.1)


def test_mock_client_default_heuristic_scores_a_clean_eligible_case_high():
    client = MockJevClient()
    result = client.score(
        {
            "risk_report": {"blocks_automatic_refund": False},
            "check_return_policy": {"eligible": True, "requires_escalation": False},
        }
    )
    assert result.score >= 0.85
    assert result.confidence >= 0.9


def test_mock_client_default_heuristic_scores_an_ineligible_case_low():
    client = MockJevClient()
    result = client.score({"risk_report": {}, "check_return_policy": {"eligible": False}})
    assert result.score <= 0.15


def test_mock_client_default_heuristic_is_ambiguous_when_escalation_required():
    client = MockJevClient()
    result = client.score(
        {
            "risk_report": {"blocks_automatic_refund": False},
            "check_return_policy": {"eligible": True, "requires_escalation": True},
        }
    )
    assert result.confidence < 0.9  # not confident enough to gate on


def test_mock_client_accepts_an_injectable_scorer():
    forced = JevScore(score=0.42, confidence=0.99, raw={"forced": True})
    client = MockJevClient(scorer=lambda query: forced)

    assert client.score({}) is forced


def test_get_jev_client_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("JEV_MODE", raising=False)
    assert isinstance(get_jev_client(), MockJevClient)


def test_get_jev_client_mode_mock_explicit(monkeypatch):
    monkeypatch.setenv("JEV_MODE", "mock")
    assert isinstance(get_jev_client(), MockJevClient)


def test_get_jev_client_rejects_an_unknown_mode(monkeypatch):
    monkeypatch.setenv("JEV_MODE", "nonsense")
    with pytest.raises(ValueError):
        get_jev_client()


def test_jev_api_client_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        JevAPIClient()


def test_jev_api_client_raises_a_clear_error_without_the_package(monkeypatch):
    # typesafe_sdk is not installed in this environment (unverified package,
    # see jev_client.py's module docstring) -- constructing with a key still
    # must fail clearly rather than with a bare ImportError.
    monkeypatch.setenv("JEV_API_KEY", "fake-key-for-test")
    with pytest.raises(RuntimeError, match="typesafe-sdk"):
        JevAPIClient()


def test_get_jev_client_mode_live_surfaces_the_same_missing_package_error(monkeypatch):
    monkeypatch.setenv("JEV_MODE", "live")
    monkeypatch.setenv("JEV_API_KEY", "fake-key-for-test")
    with pytest.raises(RuntimeError, match="typesafe-sdk"):
        get_jev_client()
