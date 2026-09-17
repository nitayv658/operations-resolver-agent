"""Tests for resolver_agent_crewai.crew.schemas -- the Pydantic mirrors of
submit_risk_report / submit_decision used as CrewAI's output_pydantic (see
schemas.py's own module docstring for why these aren't
resolver_agent.crew.schemas.RiskReport/Decision directly).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from resolver_agent_crewai.crew.schemas import CommsOutput, DecisionOutput, ResearcherOutput


def test_researcher_output_lookup_failed_needs_only_status_and_order_id():
    output = ResearcherOutput(status="LOOKUP_FAILED", order_id="ORD-2222")
    assert output.error is None
    assert output.risk_score is None


def test_researcher_output_rejects_a_status_outside_the_two_real_values():
    with pytest.raises(ValidationError):
        ResearcherOutput(status="MAYBE", order_id="ORD-1001")


def test_researcher_output_ok_accepts_every_risk_field():
    output = ResearcherOutput(
        status="OK",
        order_id="ORD-1005",
        user_id="USR-105",
        risk_score=90,
        risk_band="high",
        action_hint="escalate",
        triggered_rules=[{"rule_id": "FR-01"}],
        evidence={"order_total_usd": 480.0},
        blocks_automatic_refund=True,
        requires_security_channel=True,
        rulebook_version="1.0.0",
    )
    assert output.risk_band == "high"
    assert output.evidence == {"order_total_usd": 480.0}


def test_decision_output_does_not_carry_a_nested_risk_report():
    # Deliberate -- risk_report is attached by code afterward from the
    # already-known upstream RiskReport, not re-emitted by the model. See
    # schemas.py's module docstring.
    assert "risk_report" not in DecisionOutput.model_fields


def test_decision_output_rejects_a_refund_status_outside_the_three_real_values():
    with pytest.raises(ValidationError):
        DecisionOutput(
            order_id="ORD-1001",
            user_id="USR-101",
            verdict="ELIGIBLE",
            eligible=True,
            refund_status="MAYBE_LATER",
            requested_amount=35.0,
            applicable_policies=[],
            rationale="...",
        )


def test_decision_output_allows_omitting_approved_amount_and_refund_id():
    decision = DecisionOutput(
        order_id="ORD-1001",
        user_id="USR-101",
        verdict="ELIGIBLE",
        eligible=True,
        refund_status="ESCALATION_REQUIRED",
        requested_amount=480.0,
        applicable_policies=["POL-RET-02"],
        rationale="...",
    )
    assert decision.approved_amount is None
    assert decision.refund_id is None


def test_comms_output_is_just_the_customer_response():
    assert set(CommsOutput.model_fields.keys()) == {"customer_response"}
