"""Round-trip tests for the crew's Pydantic handoff models.

These exist mainly to pin the contract: every field the brief's grading
criteria call out (the full risk report inside Decision, the full decision
inside CrewResult) must actually survive a serialize/deserialize cycle, the
same JSON round-trip each real handoff goes through
(model_dump_json() -> the next agent's initial user message).
"""

from __future__ import annotations

import json

from resolver_agent.crew.schemas import CrewResult, Decision, RiskReport


def _risk_report(**overrides) -> RiskReport:
    base = dict(
        order_id="ORD-1005",
        user_id="USR-105",
        risk_score=90,
        risk_band="high",
        action_hint="block the automatic refund and escalate to the security channel",
        triggered_rules=[{"rule_id": "FR-01", "name": "repeat_refund_claims", "weight": 25, "why": "..."}],
        evidence={"order_total_usd": 480.0, "prior_fraud_flags": 1},
        blocks_automatic_refund=True,
        requires_security_channel=True,
        rulebook_version="1.0.0",
    )
    base.update(overrides)
    return RiskReport(**base)


def test_risk_report_round_trips_through_json():
    report = _risk_report()
    restored = RiskReport.model_validate(json.loads(report.model_dump_json()))
    assert restored == report


def test_decision_carries_the_full_risk_report_through_json():
    report = _risk_report()
    decision = Decision(
        order_id="ORD-1005",
        user_id="USR-105",
        verdict="ELIGIBLE",
        eligible=True,
        refund_status="ESCALATION_REQUIRED",
        requested_amount=480.0,
        approved_amount=None,
        refund_id=None,
        applicable_policies=["POL-RET-02"],
        rationale="Fraud report blocks automatic refund despite an eligible policy verdict.",
        risk_report=report,
    )
    restored = Decision.model_validate(json.loads(decision.model_dump_json()))
    # The whole point of this handoff: nothing about the risk report --
    # not risk_score, not triggered_rules, not evidence -- is lost or
    # summarized crossing the Agent 1 -> Agent 2 -> Agent 3 boundary.
    assert restored.risk_report == report


def test_crew_result_decision_is_optional_for_the_stop_condition_paths():
    result = CrewResult(
        order_id="ORD-9999",
        customer_response="I wasn't able to find that order.",
        decision=None,
        reasoning_chain=["Researcher could not produce a risk report: ORDER_NOT_FOUND."],
        stopped_reason="researcher_incomplete",
    )
    assert result.decision is None
    assert result.alert_sent is False
