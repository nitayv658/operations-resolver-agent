#!/usr/bin/env python3
"""Resolve one ad-hoc support ticket and pretty-print the structured result.

Usage:
    python3 run_ticket.py "Hi, I'm Maya. My earbuds from order ORD-1001 ..."

Requires ANTHROPIC_API_KEY to be set (in the environment or in a .env file --
see .env.example).
"""

from __future__ import annotations

import json
import sys

from resolver_agent import ResolverAgent


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} \"<ticket text>\"", file=sys.stderr)
        return 2

    ticket_text = sys.argv[1]
    agent = ResolverAgent()
    result = agent.resolve(ticket_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("_validation_warnings"):
        print("\n--- validation warnings ---", file=sys.stderr)
        for warning in result["_validation_warnings"]:
            print(f"  ! {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
