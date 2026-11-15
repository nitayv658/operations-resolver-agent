"""Tests for CrewAIOperationsCrew that don't require a live model call --
per-agent model resolution (mirrors resolver_agent.crew.orchestrator's own
resolution chain) and the Researcher OK-field backfill helper.

Full end-to-end guardrail parity (the ORD-1005 fraud override, the
decision-incomplete fallback alert, the cap-leak correction, Comms's
anti-overstatement checks) is verified live via
scripts/run_crew_scenarios_crewai.py against the existing 6-scenario
baseline instead -- same reasoning as resolver_agent_crewai/'s own test
suite: CrewAI's own LLM-call boundary is too version-fragile to script
reliably.
"""

from __future__ import annotations

from resolver_agent_crewai.crew.orchestrator import (
    _backfill_researcher_ok_fields,
    CrewAIOperationsCrew,
)


def test_default_construction_resolves_researcher_and_decision_to_the_shared_model():
    crew = CrewAIOperationsCrew(model="claude-opus-4")
    assert crew.researcher_model == "anthropic/claude-opus-4"
    assert crew.decision_model == "anthropic/claude-opus-4"


def test_default_construction_does_not_chain_comms_to_the_shared_model():
    # Comms is deliberately NOT chained to `model` -- see
    # resolver_agent/crew/orchestrator.py's own DEFAULT_COMMS_MODEL docstring.
    crew = CrewAIOperationsCrew(model="claude-opus-4")
    assert crew.comms_model == "anthropic/claude-haiku-4-5-20251001"


def test_explicit_per_agent_model_kwargs_override_the_shared_model():
    crew = CrewAIOperationsCrew(
        model="claude-opus-4",
        researcher_model="claude-sonnet-5",
        decision_model="claude-sonnet-5",
        comms_model="claude-opus-4",
    )
    assert crew.researcher_model == "anthropic/claude-sonnet-5"
    assert crew.decision_model == "anthropic/claude-sonnet-5"
    assert crew.comms_model == "anthropic/claude-opus-4"


def test_env_var_overrides_the_shared_model_when_no_explicit_kwarg(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL_RESEARCHER", "claude-haiku-4-5-20251001")
    crew = CrewAIOperationsCrew(model="claude-opus-4")
    assert crew.researcher_model == "anthropic/claude-haiku-4-5-20251001"
    # Decision is unaffected by the Researcher-specific env var.
    assert crew.decision_model == "anthropic/claude-opus-4"


def test_explicit_kwarg_beats_the_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL_DECISION", "claude-haiku-4-5-20251001")
    crew = CrewAIOperationsCrew(decision_model="claude-opus-4")
    assert crew.decision_model == "anthropic/claude-opus-4"


def test_backfill_leaves_a_lookup_failed_report_untouched():
    raw = {"status": "LOOKUP_FAILED", "order_id": "ORD-1001", "error": {"error": "ORDER_NOT_FOUND"}}
    assert _backfill_researcher_ok_fields(raw) == raw


def test_backfill_fills_every_missing_ok_field_with_a_type_safe_placeholder():
    raw = {"status": "OK", "order_id": "ORD-1001"}  # every OK field omitted
    filled = _backfill_researcher_ok_fields(raw)

    assert filled["status"] == "OK"
    assert filled["order_id"] == "ORD-1001"
    assert isinstance(filled["user_id"], str) and filled["user_id"]
    assert isinstance(filled["risk_score"], int) and not isinstance(filled["risk_score"], bool)
    assert filled["risk_band"] in ("low", "medium", "high")
    assert isinstance(filled["triggered_rules"], list)
    assert isinstance(filled["evidence"], dict)
    assert isinstance(filled["blocks_automatic_refund"], bool)
    assert isinstance(filled["requires_security_channel"], bool)
    assert isinstance(filled["rulebook_version"], str)


def test_backfill_leaves_correctly_typed_fields_alone():
    raw = {
        "status": "OK",
        "order_id": "ORD-1001",
        "user_id": "USR-101",
        "risk_score": 90,
        "risk_band": "high",
        "action_hint": "escalate",
        "triggered_rules": [{"rule_id": "FR-01"}],
        "evidence": {"order_total_usd": 480.0},
        "blocks_automatic_refund": True,
        "requires_security_channel": True,
        "rulebook_version": "1.0.0",
    }
    assert _backfill_researcher_ok_fields(dict(raw)) == raw


def test_default_comms_model_when_shared_model_is_not_anthropic_should_follow_it(monkeypatch):
    for var in ("ANTHROPIC_MODEL_RESEARCHER", "ANTHROPIC_MODEL_DECISION", "ANTHROPIC_MODEL_COMMS"):
        monkeypatch.delenv(var, raising=False)

    crew = CrewAIOperationsCrew(model="openai/gpt-4o")

    assert crew.researcher_model == "openai/gpt-4o"
    assert crew.decision_model == "openai/gpt-4o"
    assert crew.comms_model == "openai/gpt-4o"


def _patch_stage_boundary(monkeypatch, exc):
    import resolver_agent_crewai.crew.orchestrator as orchestrator

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

    class _FailingCrew(_Stub):
        def kickoff(self):
            raise exc

    monkeypatch.setattr(orchestrator, "Agent", _Stub)
    monkeypatch.setattr(orchestrator, "Task", _Stub)
    monkeypatch.setattr(orchestrator, "Crew", _FailingCrew)


def _openai_error():
    import httpx
    import openai

    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1/chat"))


def test_run_researcher_when_a_non_anthropic_api_error_is_raised_should_report_api_error(monkeypatch):
    _patch_stage_boundary(monkeypatch, _openai_error())

    result = CrewAIOperationsCrew(model="openai/gpt-4o")._run_researcher("ticket", "case1")

    assert result.report is None
    assert result.stopped_reason == "api_error"
    assert result.error["error"] == "MODEL_API_ERROR"


def test_run_decision_when_a_non_anthropic_api_error_is_raised_should_report_api_error(monkeypatch):
    from types import SimpleNamespace

    _patch_stage_boundary(monkeypatch, _openai_error())
    risk_report = SimpleNamespace(model_dump_json=lambda: "{}", order_id="ORD-1001")

    result = CrewAIOperationsCrew(model="openai/gpt-4o")._run_decision(risk_report, "case1")

    assert result.decision is None
    assert result.stopped_reason == "api_error"
    assert result.error["error"] == "MODEL_API_ERROR"


def test_run_comms_when_a_non_anthropic_api_error_is_raised_should_fall_back_to_the_safe_reply(monkeypatch):
    from types import SimpleNamespace

    _patch_stage_boundary(monkeypatch, _openai_error())
    decision = SimpleNamespace(model_dump_json=lambda: "{}", refund_status="REJECTED")

    result = CrewAIOperationsCrew(model="openai/gpt-4o")._run_comms(decision, "case1")

    assert result.stopped_reason == "api_error"
    assert result.customer_response
    assert result.alert_sent is False
