"""Tests for the stage-9 "trigger workflow" side effect.

Pure unit tests against escalation_workflow directly -- no ScriptedClient or
model turns involved, since trigger_workflow only ever looks at an already-
built resolution dict.
"""

from __future__ import annotations

import json

import pytest

from resolver_agent.escalation_workflow import build_escalation_record, trigger_workflow


def _resolution(decision, **overrides):
    base = {
        "reasoning_chain": ["ORD-1002 total is $150.00, above the $50 Standard cap."],
        "action_taken": {
            "tools_called": ["get_order_details", "process_refund"],
            "decision": decision,
            "refund_amount": None,
            "refund_id": None,
        },
        "customer_response": "This has been escalated to our team.",
        "_corrections": [],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("decision", ["AUTO_REFUND_APPROVED", "REJECTED"])
def test_trigger_workflow_when_decision_is_terminal_should_be_a_noop(tmp_path, decision):
    path = tmp_path / "queue.jsonl"

    record = trigger_workflow(_resolution(decision), "case123", queue_path=path)

    assert record is None
    assert not path.exists()


@pytest.mark.parametrize("decision", ["ESCALATION_REQUIRED", "CANNOT_RESOLVE"])
def test_trigger_workflow_when_decision_needs_a_human_should_write_a_record(tmp_path, decision):
    path = tmp_path / "queue.jsonl"

    record = trigger_workflow(_resolution(decision), "case123", queue_path=path)

    assert record is not None
    assert record["decision"] == decision
    assert record["case_id"] == "case123"

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written == record


def test_trigger_workflow_should_append_not_overwrite_across_multiple_cases(tmp_path):
    path = tmp_path / "queue.jsonl"

    trigger_workflow(_resolution("ESCALATION_REQUIRED"), "case-a", queue_path=path)
    trigger_workflow(_resolution("CANNOT_RESOLVE"), "case-b", queue_path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["case_id"] == "case-a"
    assert json.loads(lines[1])["case_id"] == "case-b"


def test_trigger_workflow_should_create_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "queue.jsonl"

    trigger_workflow(_resolution("ESCALATION_REQUIRED"), "case123", queue_path=path)

    assert path.exists()


def test_trigger_workflow_record_should_never_include_customer_response():
    # Same privacy stance as logging_utils.py: raw ticket text and
    # customer_response can carry a customer's name and are never persisted
    # outside the immediate resolve() return value.
    record = build_escalation_record(_resolution("ESCALATION_REQUIRED"), "case123")

    assert "customer_response" not in record
    assert "ticket_text" not in record


def test_trigger_workflow_should_use_injected_writer_instead_of_touching_disk(tmp_path):
    written = []

    def fake_writer(record, path):
        written.append((record, path))

    result = trigger_workflow(
        _resolution("ESCALATION_REQUIRED"),
        "case123",
        queue_path=tmp_path / "unused.jsonl",
        writer=fake_writer,
    )

    assert result is not None
    assert written == [(result, tmp_path / "unused.jsonl")]
    assert not (tmp_path / "unused.jsonl").exists()
