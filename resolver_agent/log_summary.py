"""Aggregate resolver_agent's structured JSON logs into a plain-text summary.

logging_utils.py already gives every event the right categorical shape
(a dotted ``event`` name plus structured fields, one JSON object per line) --
what was missing was anything that actually aggregated them. This is
deliberately the "natural first step" toward real monitoring described in
the project's own README, not a replacement for shipping logs to a real
platform (CloudWatch/Datadog/ELK/etc.) and alerting there -- it just makes
"how many cases hit X this run" answerable from a log file without one.

No I/O in this module: :func:`summarize` takes any ``Iterable[str]`` (a file,
``sys.stdin``, a plain list) and returns a plain dataclass, so it's testable
without touching disk. ``summarize_logs.py`` at the repo root is the thin CLI
wrapper.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

# Events worth watching for a rising rate specifically -- each one means
# the agent silently degraded or a security-relevant denial fired, as
# opposed to a normal business decision. Pulled directly from the
# log_event() call sites in agent.py/tool_loop.py/escalation_workflow.py;
# see test_log_summary.py's own pinning test against the real event names.
DEGRADATION_EVENTS = (
    "agent.api_error",
    "agent.fallback_resolution_used",
    "agent.resolution_corrected",
    "agent.unauthorized_tool_result_denied",
    "escalation_workflow.webhook_delivery_failed",
    "tool_loop.max_iterations_reached",
)


@dataclass
class LogSummary:
    total_lines: int
    parsed_lines: int
    total_cases: int
    event_counts: Dict[str, int] = field(default_factory=dict)
    event_case_counts: Dict[str, int] = field(default_factory=dict)


def summarize(lines: Iterable[str]) -> LogSummary:
    """Parse ``lines`` as resolver_agent's JSON log format and aggregate.

    A line that isn't valid JSON, or is valid JSON but has no ``event`` key
    (not one of our structured log lines), is silently skipped -- this is
    what lets a caller pipe combined stdout+stderr straight in (e.g.
    ``run_scenarios.py``'s own prose output) without pre-filtering.
    """
    total_lines = 0
    parsed_lines = 0
    event_counts: Counter = Counter()
    event_cases: Dict[str, Set[str]] = defaultdict(set)
    all_cases: Set[str] = set()

    for line in lines:
        total_lines += 1
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or "event" not in record:
            continue

        parsed_lines += 1
        event = record["event"]
        event_counts[event] += 1

        case_id = record.get("case_id")
        if case_id is not None:
            all_cases.add(case_id)
            event_cases[event].add(case_id)

    return LogSummary(
        total_lines=total_lines,
        parsed_lines=parsed_lines,
        total_cases=len(all_cases),
        event_counts=dict(event_counts),
        event_case_counts={name: len(cases) for name, cases in event_cases.items()},
    )


def format_summary(summary: LogSummary) -> str:
    lines = [
        f"Parsed {summary.parsed_lines}/{summary.total_lines} lines as structured "
        f"log events across {summary.total_cases} distinct case(s).",
        "",
    ]

    if summary.parsed_lines == 0:
        lines.append("No structured log events found.")
        return "\n".join(lines)

    lines.append("Degradation signals (worth alerting on a rising rate):")
    any_signal = False
    for event in DEGRADATION_EVENTS:
        count = summary.event_counts.get(event)
        if not count:
            continue
        any_signal = True
        cases = summary.event_case_counts.get(event, count)
        pct = f" ({100 * cases / summary.total_cases:.0f}% of cases)" if summary.total_cases else ""
        lines.append(f"  {event}: {count} line(s), {cases} case(s){pct}")
    if not any_signal:
        lines.append("  none")
    lines.append("")

    lines.append("All events:")
    for event, count in sorted(summary.event_counts.items(), key=lambda item: -item[1]):
        cases = summary.event_case_counts.get(event)
        case_suffix = f", {cases} case(s)" if cases is not None else ""
        lines.append(f"  {event}: {count} line(s){case_suffix}")

    return "\n".join(lines)
