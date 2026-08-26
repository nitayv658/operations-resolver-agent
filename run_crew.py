#!/usr/bin/env python3
"""Resolve one ad-hoc support ticket through the Part 2 multi-agent crew and
pretty-print the result.

Usage:
    python3 run_crew.py "This is Ronen, order ORD-1005. The tablet screen ..."

Requires ANTHROPIC_API_KEY to be set (in the environment or in a .env file --
see .env.example). Set SLACK_WEBHOOK_URL to also POST any alert to a real
Slack incoming webhook; otherwise it's written to
starter-kit/outbox/alerts.jsonl only.

Structured logs go to stderr as JSON (one object per line), controlled by
LOG_LEVEL (default WARNING). The crew's result is the only thing printed to
stdout.
"""

from __future__ import annotations

import json
import sys

from resolver_agent.crew import OperationsCrew
from resolver_agent.logging_utils import configure_logging


def main() -> int:
    configure_logging()

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} \"<ticket text>\"", file=sys.stderr)
        return 2

    ticket_text = sys.argv[1]
    crew = OperationsCrew()
    result = crew.handle_ticket(ticket_text)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
