#!/usr/bin/env python3
"""Run the CrewAI port against the same 10 scenarios as scripts/run_scenarios.py.

The CrewAI-framework counterpart to scripts/run_scenarios.py -- reuses the
exact same SCENARIOS list (imported, not copied) so the two implementations'
decisions can be diffed 1:1 on identical tickets. Requires the CrewAI extras
(see requirements-crewai.txt / .venv-crewai) and ANTHROPIC_API_KEY.

Usage:
    python3 scripts/run_scenarios_crewai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))  # for `from run_scenarios import SCENARIOS`
sys.path.insert(0, str(_SCRIPTS_DIR.parent))  # resolver_agent_crewai/ is one level up

from resolver_agent.logging_utils import configure_logging  # noqa: E402
from resolver_agent_crewai import CrewResolverAgent  # noqa: E402
from run_scenarios import SCENARIOS  # noqa: E402


def main() -> int:
    configure_logging()
    agent = CrewResolverAgent()
    failures = 0

    for scenario in SCENARIOS:
        print(f"\n=== Scenario {scenario['id']} -- {scenario['title']} ===")
        print(f"ticket: {scenario['ticket']}")
        try:
            result = agent.resolve(scenario["ticket"])
        except Exception as exc:  # noqa: BLE001 -- report, don't abort the run
            print(f"  ERROR: agent raised {exc!r}")
            failures += 1
            continue

        decision = (result.get("action_taken") or {}).get("decision")
        expected = scenario["expected"]
        ok = decision == expected
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] decision={decision!r} expected={expected!r} case_id={result.get('_case_id')!r}")
        print(f"  tools_called: {(result.get('action_taken') or {}).get('tools_called')}")
        print(f"  customer_response: {result.get('customer_response')}")

        warnings = result.get("_validation_warnings") or []
        if warnings:
            print("  validation warnings:")
            for warning in warnings:
                print(f"    ! {warning}")

        if not ok or warnings:
            failures += 1

    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios matched the expected decision cleanly.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
