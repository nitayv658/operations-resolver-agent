#!/usr/bin/env python3
"""Run the crew against the headline Stage 2 scenarios plus a Part 1
regression spot-check.

Same relationship to starter-kit/examples/verify_scenarios.py as
run_scenarios.py has for Part 1: that script checks the deterministic
data/rule engines are self-consistent (run it first); this one checks
whether the *crew* -- three real model calls per ticket -- reaches the
right outcome, including whether an alert actually lands in the outbox
exactly when (and only when) it should.

Usage:
    python3 run_crew_scenarios.py
"""

from __future__ import annotations

import sys

from resolver_agent.crew import OperationsCrew
from resolver_agent.logging_utils import configure_logging

sys.path.insert(0, "starter-kit")
import multi_agent_tools as mat  # noqa: E402

SCENARIOS = [
    {
        "id": "B1",
        "title": "The headline fraud case -- ELIGIBLE policy, high fraud risk",
        "ticket": (
            "This is Ronen, order ORD-1005. The tablet screen was smashed on "
            "arrival. Refund me the full 480 dollars, this keeps happening."
        ),
        "expected_refund_status": "ESCALATION_REQUIRED",
        "expect_alert": True,
    },
    {
        "id": "B2",
        "title": "New account, high value, item never arrived",
        "ticket": (
            "I ordered a laptop (ORD-1012), the box arrived but it was "
            "empty. I need the 890 dollars back."
        ),
        "expected_refund_status": "ESCALATION_REQUIRED",
        "expect_alert": True,
    },
    {
        "id": "B3",
        "title": "Clean case -- no escalation, no alert",
        "ticket": (
            "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked "
            "right out of the box. Can you sort this out?"
        ),
        "expected_refund_status": "APPROVED",
        "expect_alert": False,
    },
    {
        "id": "P1-2",
        "title": "Part 1 regression -- authority breach still escalates",
        "ticket": (
            "Order ORD-1002. The espresso machine is dented and leaking. I "
            "paid 150 dollars for this. I want my money back today."
        ),
        "expected_refund_status": "ESCALATION_REQUIRED",
        "expect_alert": True,
    },
    {
        "id": "P1-3",
        "title": "Part 1 regression -- outside the return window still rejects",
        "ticket": (
            "I ordered a backpack back at the end of May (ORD-1003) and "
            "I've changed my mind, I'd like to return it."
        ),
        "expected_refund_status": "REJECTED",
        # get_escalation_route routes any OUTSIDE_RETURN_WINDOW verdict to
        # #support-tier2 (priority 3: "over cap but under $250, or a
        # rejected claim") -- a rejected claim still gets tracked, just not
        # paged as fraud.
        "expect_alert": True,
    },
    {
        "id": "P1-8",
        "title": "Part 1 regression -- non-returnable category still rejects",
        "ticket": "ORD-1008, I bought a gift card by accident. Please refund it.",
        "expected_refund_status": "REJECTED",
        "expect_alert": True,  # NON_RETURNABLE_CATEGORY -> #support-tier2, same as P1-3 above
    },
]


def main() -> int:
    configure_logging()
    crew = OperationsCrew()
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
