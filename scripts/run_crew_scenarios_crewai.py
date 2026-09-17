#!/usr/bin/env python3
"""Run the CrewAI port of Part 2's crew against the same 6 scenarios as
scripts/run_crew_scenarios.py.

The CrewAI-framework counterpart to scripts/run_crew_scenarios.py -- reuses
the exact same SCENARIOS list (imported, not copied) so the two
implementations' decisions and alert-dispatch behavior can be diffed 1:1 on
identical tickets. Requires the CrewAI extras (see requirements-crewai.txt /
.venv-crewai) and ANTHROPIC_API_KEY.

Usage:
    python3 scripts/run_crew_scenarios_crewai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))  # for `from run_crew_scenarios import SCENARIOS`
sys.path.insert(0, str(_SCRIPTS_DIR.parent))  # resolver_agent_crewai/ is one level up

from resolver_agent.logging_utils import configure_logging  # noqa: E402
from resolver_agent_crewai.crew import CrewAIOperationsCrew  # noqa: E402
from run_crew_scenarios import SCENARIOS  # noqa: E402

# starter-kit/ is already on sys.path as a side effect of the imports above.
import multi_agent_tools as mat  # noqa: E402


def main() -> int:
    configure_logging()
    crew = CrewAIOperationsCrew()
    failures = 0

    for scenario in SCENARIOS:
        print(f"\n=== Scenario {scenario['id']} -- {scenario['title']} ===")
        print(f"ticket: {scenario['ticket']}")

        outbox_before = len(mat.read_outbox())
        try:
            result = crew.handle_ticket(scenario["ticket"])
        except Exception as exc:  # noqa: BLE001 -- report, don't abort the run
            print(f"  ERROR: crew raised {exc!r}")
            failures += 1
            continue
        outbox_after = len(mat.read_outbox())
        alert_written = outbox_after > outbox_before

        status = result.decision.refund_status if result.decision else result.stopped_reason
        expected = scenario["expected_refund_status"]
        status_ok = status == expected
        alert_ok = alert_written == scenario["expect_alert"]
        ok = status_ok and alert_ok

        print(f"  [{'ok  ' if ok else 'FAIL'}] refund_status={status!r} expected={expected!r}")
        print(
            f"  alert_written={alert_written} expected={scenario['expect_alert']} "
            f"(outbox {outbox_before} -> {outbox_after})"
        )
        print(f"  customer_response: {result.customer_response}")
        if not ok:
            failures += 1

    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios matched.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
