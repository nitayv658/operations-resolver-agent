#!/usr/bin/env python3
"""Resolve one ad-hoc support ticket and pretty-print the structured result.

Usage:
    python3 run_ticket.py "Hi, I'm Maya. My earbuds from order ORD-1001 ..."
    python3 run_ticket.py "..." USR-101   # bind to a requester identity --
                                           # any record naming a different
                                           # owner gets denied, not disclosed

Requires ANTHROPIC_API_KEY to be set (in the environment or in a .env file --
see .env.example).

Structured logs go to stderr as JSON (one object per line), controlled by
LOG_LEVEL (default WARNING -- a clean run prints nothing to stderr). The
agent's actual result is the only thing printed to stdout.
"""

from __future__ import annotations

import json
import sys

from resolver_agent import ResolverAgent
from resolver_agent.logging_utils import configure_logging


def main() -> int:
    configure_logging()

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} \"<ticket text>\" [requester_user_id]", file=sys.stderr)
        return 2

    ticket_text = sys.argv[1]
    requester_user_id = sys.argv[2] if len(sys.argv) > 2 else None
    agent = ResolverAgent()
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
