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
