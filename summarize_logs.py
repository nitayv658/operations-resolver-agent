#!/usr/bin/env python3
"""Aggregate resolver_agent's structured JSON logs into a plain-text summary.

Usage:
    python3 summarize_logs.py path/to/log.jsonl
    python3 summarize_logs.py -                          # read stdin explicitly
    python3 run_ticket.py "..." 2> run.log && python3 summarize_logs.py run.log
    LOG_LEVEL=INFO python3 run_scenarios.py 2>&1 | python3 summarize_logs.py

No file path (or ``-``) reads from stdin. Non-JSON lines (e.g.
run_scenarios.py's own "=== Scenario N ===" prose) are silently skipped, so
piping combined stdout+stderr straight in works without pre-filtering.

LOG_LEVEL defaults to WARNING (see logging_utils.configure_logging) -- some
events this summarizes (agent.case_resolved, agent.workflow_triggered) are
only emitted at INFO, so set LOG_LEVEL=INFO when generating the log you want
to summarize if you want cases-resolved/workflow-triggered counts included.

This is a local aggregate over whatever log file/stream you hand it -- not a
replacement for shipping logs to a real monitoring platform and alerting
there (see issue #6). It's the "natural first step" toward that.
"""

from __future__ import annotations

import sys

from resolver_agent.log_summary import format_summary, summarize


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "-"

    if path == "-":
        lines = sys.stdin
        print(format_summary(summarize(lines)))
    else:
        with open(path, "r", encoding="utf-8") as fh:
            print(format_summary(summarize(fh)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
