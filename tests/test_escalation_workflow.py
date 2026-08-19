"""Tests for the stage-9 "trigger workflow" side effect.

Pure unit tests against escalation_workflow directly -- no ScriptedClient or
model turns involved, since trigger_workflow only ever looks at an already-
built resolution dict.
"""

from __future__ import annotations

import json
import logging
import urllib.error

import pytest

from resolver_agent import escalation_workflow
from resolver_agent.escalation_workflow import (
    WebhookDeliveryError,
    build_escalation_record,
    build_webhook_writer,
    post_webhook,
    trigger_workflow,
)


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


# --------------------------------------------------------------------------- #
# Webhook delivery -- _urlopen is monkeypatched so these never touch the real
# network. post_webhook/build_webhook_writer are tested directly; the
# ESCALATION_WEBHOOK_URL env-var wiring is tested at the trigger_workflow
# level below.
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_post_webhook_when_delivery_succeeds_should_post_the_record_as_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["content_type"] = request.get_header("Content-type")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(escalation_workflow, "_urlopen", fake_urlopen)

    post_webhook({"case_id": "abc123", "decision": "ESCALATION_REQUIRED"}, "https://ops.example.com/hook")

    assert captured["url"] == "https://ops.example.com/hook"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    assert captured["body"] == {"case_id": "abc123", "decision": "ESCALATION_REQUIRED"}


def test_post_webhook_when_delivery_fails_should_raise_webhook_delivery_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(escalation_workflow, "_urlopen", fake_urlopen)

    with pytest.raises(WebhookDeliveryError, match="connection refused"):
        post_webhook({"case_id": "abc123"}, "https://ops.example.com/hook")


def test_webhook_writer_when_delivery_succeeds_should_not_touch_the_fallback_file(tmp_path, monkeypatch):
    monkeypatch.setattr(escalation_workflow, "_urlopen", lambda request, timeout: _FakeResponse())
    writer = build_webhook_writer("https://ops.example.com/hook")
    path = tmp_path / "fallback.jsonl"

    writer({"case_id": "abc123"}, path)

    assert not path.exists()


def test_webhook_writer_when_delivery_fails_should_fall_back_to_the_local_file_and_log(tmp_path, monkeypatch, caplog):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(escalation_workflow, "_urlopen", fake_urlopen)
    writer = build_webhook_writer("https://ops.example.com/hook")
    path = tmp_path / "fallback.jsonl"

    with caplog.at_level(logging.WARNING, logger="resolver_agent"):
        writer({"case_id": "abc123"}, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"case_id": "abc123"}

    records = [r for r in caplog.records if r.getMessage() == "escalation_workflow.webhook_delivery_failed"]
    assert len(records) == 1
    assert records[0].fields["case_id"] == "abc123"


def test_trigger_workflow_when_webhook_url_env_var_is_set_should_use_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", "https://ops.example.com/hook")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _FakeResponse()

    monkeypatch.setattr(escalation_workflow, "_urlopen", fake_urlopen)
    path = tmp_path / "unused.jsonl"

    record = trigger_workflow(_resolution("ESCALATION_REQUIRED"), "case123", queue_path=path)

    assert record is not None
    assert captured["url"] == "https://ops.example.com/hook"
    assert not path.exists()  # delivered successfully -- no fallback needed


def test_trigger_workflow_when_webhook_url_is_unset_should_use_the_local_file(tmp_path):
    # The conftest autouse fixture already deletes ESCALATION_WEBHOOK_URL --
    # this just makes the default explicit and pins the regression.
    path = tmp_path / "queue.jsonl"

    trigger_workflow(_resolution("ESCALATION_REQUIRED"), "case123", queue_path=path)

    assert path.exists()


# --------------------------------------------------------------------------- #
# Security review findings: (1) no scheme/TLS enforcement on the webhook URL
# -- an http:// endpoint would ship escalation records in cleartext with no
# warning; (2) reasoning_chain is freeform LLM text, never actually filtered
# for PII despite the module's stated "only structural facts" privacy intent
# -- shipping it to an arbitrary third-party webhook is a real information
# disclosure gap the local-file-only design never had. Both found during a
# senior-cyber-architect threat model review, not previously covered.
# --------------------------------------------------------------------------- #


def test_post_webhook_when_url_is_not_https_should_refuse_without_attempting_delivery(monkeypatch):
    calls = []
    monkeypatch.setattr(escalation_workflow, "_urlopen", lambda request, timeout: calls.append(1))

    with pytest.raises(WebhookDeliveryError, match="https"):
        post_webhook({"case_id": "abc123"}, "http://ops.example.com/hook")

    # must refuse before ever attempting the network call -- cleartext must
    # never be attempted, not merely warned about after the fact
    assert calls == []


@pytest.mark.parametrize("bad_url", ["http://ops.example.com/hook", "ftp://ops.example.com/hook", "not-a-url"])
def test_build_webhook_writer_when_url_is_not_https_should_fall_back_to_local_file(tmp_path, monkeypatch, bad_url):
    # _urlopen is monkeypatched to fail the test if it's ever called -- the
    # scheme check must reject these before any network attempt, not rely
    # on the real network call happening to fail.
    def fail_if_called(request, timeout):
        raise AssertionError("must not attempt delivery for a non-HTTPS/malformed URL")

    monkeypatch.setattr(escalation_workflow, "_urlopen", fail_if_called)
    writer = build_webhook_writer(bad_url)
    path = tmp_path / "fallback.jsonl"

    writer({"case_id": "abc123"}, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"case_id": "abc123"}


def test_build_webhook_writer_should_exclude_reasoning_chain_from_the_delivered_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(escalation_workflow, "_urlopen", fake_urlopen)
    writer = build_webhook_writer("https://ops.example.com/hook")
    record = build_escalation_record(_resolution("ESCALATION_REQUIRED"), "case123")
    assert record["reasoning_chain"]  # sanity: the source record does have it

    writer(record, escalation_workflow.DEFAULT_QUEUE_PATH)

    assert "reasoning_chain" not in captured["body"]
    # everything else (the actually structural fields) still makes it through
    assert captured["body"]["case_id"] == "case123"
    assert captured["body"]["decision"] == "ESCALATION_REQUIRED"


def test_build_webhook_writer_when_delivery_fails_should_fall_back_with_the_full_record(tmp_path, monkeypatch):
    # The local file is an internal artifact, same trust boundary as this
    # module's own local-file default -- unlike the webhook egress, it
    # keeps reasoning_chain, since nothing here leaves the process.
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(escalation_workflow, "_urlopen", fake_urlopen)
    writer = build_webhook_writer("https://ops.example.com/hook")
    path = tmp_path / "fallback.jsonl"
    record = build_escalation_record(_resolution("ESCALATION_REQUIRED"), "case123")

    writer(record, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    written = json.loads(lines[0])
    assert written["reasoning_chain"] == record["reasoning_chain"]


def test_build_webhook_writer_when_delivery_fails_should_never_log_the_url_query_string(tmp_path, monkeypatch, caplog):
    # Webhook auth is commonly a signed token in the query string
    # (?token=...). A delivery failure must not write that token to logs --
    # only scheme/host/path, never query or fragment.
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(escalation_workflow, "_urlopen", fake_urlopen)
    secret_url = "https://ops.example.com/hook?token=super-secret-value"
    writer = build_webhook_writer(secret_url)
    path = tmp_path / "fallback.jsonl"

    with caplog.at_level(logging.WARNING, logger="resolver_agent"):
        writer({"case_id": "abc123"}, path)

    for record in caplog.records:
        text = record.getMessage() + json.dumps(getattr(record, "fields", {}))
        assert "super-secret-value" not in text
