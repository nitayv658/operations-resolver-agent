#!/usr/bin/env python3
"""Resolve one ad-hoc support ticket through the CrewAI port of Part 2's
multi-agent crew, and pretty-print the result.

The CrewAI-framework counterpart to scripts/run_crew.py -- same usage, same
output shape (resolver_agent.crew.schemas.CrewResult), pointed at
CrewAIOperationsCrew (resolver_agent_crewai/crew/) instead of the hand-rolled
OperationsCrew. Requires the CrewAI extras (see requirements-crewai.txt,
which needs Python >=3.10 -- run this from a separate environment, e.g.
`.venv-crewai`).

Usage:
    python3 scripts/run_crew_crewai.py "This is Ronen, order ORD-1005. The tablet screen ..."

Requires ANTHROPIC_API_KEY to be set (in the environment or in a .env file --
see .env.example). Set SLACK_WEBHOOK_URL to also POST any alert to a real
Slack incoming webhook; otherwise it's written to
starter-kit/outbox/alerts.jsonl only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # resolver_agent_crewai/ is one level up

from resolver_agent.logging_utils import configure_logging  # noqa: E402
from resolver_agent_crewai.crew import CrewAIOperationsCrew  # noqa: E402


def main() -> int:
    configure_logging()

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} \"<ticket text>\"", file=sys.stderr)
        return 2

    ticket_text = sys.argv[1]
    crew = CrewAIOperationsCrew()
    result = crew.handle_ticket(ticket_text)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
