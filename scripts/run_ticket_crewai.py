#!/usr/bin/env python3
"""Resolve one ad-hoc support ticket with the CrewAI port, and pretty-print
the structured result.

The CrewAI-framework counterpart to scripts/run_ticket.py -- same usage,
same output shape, different implementation (resolver_agent_crewai instead
of resolver_agent). Requires the CrewAI extras (see requirements-crewai.txt,
which needs Python >=3.10 -- this repo's main .venv is 3.9, so run this from
a separate environment, e.g. `.venv-crewai`).

Usage:
    python3 scripts/run_ticket_crewai.py "Hi, I'm Maya. My earbuds from order ORD-1001 ..."
    python3 scripts/run_ticket_crewai.py "..." USR-101   # bind to a requester identity

Requires ANTHROPIC_API_KEY to be set (in the environment or in a .env file --
see .env.example).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # resolver_agent_crewai/ is one level up

from resolver_agent.logging_utils import configure_logging  # noqa: E402
from resolver_agent_crewai import CrewResolverAgent  # noqa: E402


def main() -> int:
    configure_logging()

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} \"<ticket text>\" [requester_user_id]", file=sys.stderr)
        return 2

    ticket_text = sys.argv[1]
    requester_user_id = sys.argv[2] if len(sys.argv) > 2 else None
    agent = CrewResolverAgent()
    result = agent.resolve(ticket_text, requester_user_id=requester_user_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("_validation_warnings"):
        print("\n--- validation warnings (what the model got wrong) ---", file=sys.stderr)
        for warning in result["_validation_warnings"]:
            print(f"  ! {warning}", file=sys.stderr)

    if result.get("_corrections"):
        print("\n--- corrections (what was overridden before returning) ---", file=sys.stderr)
        for correction in result["_corrections"]:
            print(f"  > {correction}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
