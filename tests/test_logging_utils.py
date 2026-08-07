"""Tests for logging_utils.py -- the structured-logging foundation."""

from __future__ import annotations

import json
import logging

from resolver_agent.logging_utils import JsonFormatter, configure_logging, get_logger, log_event


def test_log_event_when_called_should_attach_fields_to_the_record(caplog):
    logger = get_logger("resolver_agent.test")
    with caplog.at_level(logging.INFO, logger="resolver_agent.test"):
        log_event(logger, logging.INFO, "case.resolved", case_id="abc123", decision="AUTO_REFUND_APPROVED")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "case.resolved"
    assert record.fields == {"case_id": "abc123", "decision": "AUTO_REFUND_APPROVED"}


def test_json_formatter_when_record_has_fields_should_render_them_as_top_level_keys():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="resolver_agent.agent",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="case.validation_warning",
        args=(),
        exc_info=None,
    )
    record.fields = {"case_id": "abc123", "warning_count": 1}

    payload = json.loads(formatter.format(record))
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "resolver_agent.agent"
    assert payload["event"] == "case.validation_warning"
    assert payload["case_id"] == "abc123"
    assert payload["warning_count"] == 1


def test_json_formatter_when_record_has_no_fields_should_still_render_cleanly():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="resolver_agent.tool_loop",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="tool_loop.tool_executed",
        args=(),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert payload == {
        "level": "DEBUG",
        "logger": "resolver_agent.tool_loop",
        "event": "tool_loop.tool_executed",
    }


def test_configure_logging_when_called_should_attach_exactly_one_stderr_handler():
    configure_logging(level="DEBUG")
    root = logging.getLogger("resolver_agent")
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)

    # calling again must not accumulate duplicate handlers
    configure_logging(level="INFO")
    assert len(root.handlers) == 1
    assert root.level == logging.INFO


def test_configure_logging_when_no_level_given_should_default_to_warning(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger("resolver_agent").level == logging.WARNING


def test_configure_logging_when_log_level_env_var_set_should_use_it(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    configure_logging()
    assert logging.getLogger("resolver_agent").level == logging.ERROR
