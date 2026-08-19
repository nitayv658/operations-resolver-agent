"""Tests for log_summary.summarize()/format_summary() -- the aggregation
logic behind summarize_logs.py.

No I/O here: summarize() takes an iterable of lines (a real caller might
hand it an open file, sys.stdin, or a plain list, all of which are
interchangeable as an Iterable[str]) and returns a plain dataclass, so
these tests just build lists of JSON strings directly. summarize_logs.py
itself is a thin CLI wrapper around this and isn't separately tested here.
"""

from __future__ import annotations

import json

from resolver_agent.log_summary import DEGRADATION_EVENTS, format_summary, summarize


def _line(event: str, **fields) -> str:
    payload = {"level": "INFO", "logger": "resolver_agent.agent", "event": event}
    payload.update(fields)
    return json.dumps(payload)


def test_summarize_when_given_no_lines_should_report_zero_everything():
    summary = summarize([])

    assert summary.total_lines == 0
    assert summary.parsed_lines == 0
    assert summary.total_cases == 0
    assert summary.event_counts == {}
    assert summary.event_case_counts == {}


def test_summarize_should_count_each_event_and_track_distinct_cases():
    lines = [
        _line("agent.case_resolved", case_id="case-1"),
        _line("agent.case_resolved", case_id="case-2"),
        _line("agent.api_error", case_id="case-3"),
    ]

    summary = summarize(lines)

    assert summary.parsed_lines == 3
    assert summary.total_cases == 3
    assert summary.event_counts == {"agent.case_resolved": 2, "agent.api_error": 1}
    assert summary.event_case_counts == {"agent.case_resolved": 2, "agent.api_error": 1}


def test_summarize_should_deduplicate_case_ids_within_the_same_event():
    # Not something the codebase actually does today (each event fires at
    # most once per case), but the aggregation must not silently double-
    # count a case if that ever changes.
    lines = [
        _line("tool_loop.repeat_call_refused", case_id="case-1"),
        _line("tool_loop.repeat_call_refused", case_id="case-1"),
    ]

    summary = summarize(lines)

    assert summary.event_counts["tool_loop.repeat_call_refused"] == 2  # every line still counted
    assert summary.event_case_counts["tool_loop.repeat_call_refused"] == 1  # but only one distinct case


def test_summarize_should_ignore_lines_that_are_not_valid_json():
    # Real usage pipes combined stdout+stderr from run_scenarios.py, which
    # includes plain-text "=== Scenario N ===" lines interleaved with JSON
    # log lines -- those must be silently skipped, not crash the summary.
    lines = [
        "=== Scenario 1 -- Happy path ===",
        _line("agent.case_resolved", case_id="case-1"),
        "  customer_response: some prose here, not JSON",
        "",
    ]

    summary = summarize(lines)

    assert summary.total_lines == 4
    assert summary.parsed_lines == 1
    assert summary.event_counts == {"agent.case_resolved": 1}


def test_summarize_should_ignore_json_lines_that_are_not_log_events():
    # e.g. a JSON blob printed for some other reason that happens to also
    # be valid JSON but isn't one of our structured log lines.
    lines = ['{"foo": "bar"}', "[1, 2, 3]", '"just a json string"']

    summary = summarize(lines)

    assert summary.total_lines == 3
    assert summary.parsed_lines == 0


def test_summarize_should_count_events_with_no_case_id_without_crashing():
    lines = [_line("tool_loop.unknown_tool_requested")]  # this event has no case_id field

    summary = summarize(lines)

    assert summary.event_counts == {"tool_loop.unknown_tool_requested": 1}
    assert summary.total_cases == 0
    assert summary.event_case_counts == {}


def test_degradation_events_are_a_real_subset_of_what_the_codebase_actually_emits():
    # Pins DEGRADATION_EVENTS against real event names so this list can't
    # silently drift from the actual log_event() call sites.
    real_event_names = {
        "agent.api_error",
        "agent.case_resolved",
        "agent.fallback_resolution_used",
        "agent.resolution_corrected",
        "agent.unauthorized_tool_result_denied",
        "agent.validation_warnings",
        "agent.workflow_triggered",
        "escalation_workflow.webhook_delivery_failed",
        "tool_loop.max_iterations_reached",
        "tool_loop.repeat_call_refused",
        "tool_loop.tool_executed",
        "tool_loop.unknown_tool_requested",
    }
    assert set(DEGRADATION_EVENTS) <= real_event_names


def test_format_summary_should_include_degradation_signals_with_percentage_of_cases():
    lines = [
        _line("agent.case_resolved", case_id=f"case-{i}") for i in range(8)
    ] + [
        _line("agent.api_error", case_id="case-8"),
        _line("agent.api_error", case_id="case-9"),
    ]

    text = format_summary(summarize(lines))

    assert "agent.api_error" in text
    assert "2" in text  # the count
    assert "10" in text  # total distinct cases (0-9)
    assert "20" in text  # 2/10 = 20%


def test_format_summary_when_no_lines_parsed_should_say_so_not_crash():
    text = format_summary(summarize(["not json", ""]))

    assert "0" in text
