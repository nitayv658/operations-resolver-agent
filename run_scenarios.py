#!/usr/bin/env python3
"""Run the agent against the 9 scenarios from starter-kit/examples/scenarios.md.

This is the LLM-level regression check for the *agent's judgment* -- it is
not the same thing as `starter-kit/examples/verify_scenarios.py`, which only
checks that the deterministic data + rule engine (mock_services.py) itself is
self-consistent. Run that one first; if it fails, the fixtures are broken and
nothing below means anything. This script assumes the fixtures are sound and
checks whether the *agent* reaches the right decision on realistic customer
tickets, including the three required edge cases and the hallucination trap.

Scenarios 5 and 7 are each two orders in the brief (a boundary pair, and two
non-refundable statuses), so they're split into 5a/5b and 7a/7b here -- ten
runs covering the same nine numbered scenarios.

Usage:
    python3 run_scenarios.py
"""

from __future__ import annotations

import json
import sys

from resolver_agent import ResolverAgent
from resolver_agent.logging_utils import configure_logging

SCENARIOS = [
    {
        "id": "1",
        "title": "Happy path -- VIP, damaged item, under the cap",
        "ticket": (
            "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked "
            "right out of the box. I've been shopping with you for years, "
            "can you sort this out?"
        ),
        "expected": "AUTO_REFUND_APPROVED",
    },
    {
        "id": "2",
        "title": "Authority breach -- damaged item, above the cap",
        "ticket": (
            "Order ORD-1002. The espresso machine is dented and leaking. I "
            "paid 150 dollars for this. I want my money back today."
        ),
        "expected": "ESCALATION_REQUIRED",
    },
    {
        "id": "3",
        "title": "Window breach -- 60 days after delivery",
        "ticket": (
            "I ordered a backpack back at the end of May (ORD-1003) and "
            "I've changed my mind, I'd like to return it."
        ),
        "expected": "REJECTED",
    },
    {
        "id": "4",
        "title": "Non-returnable category -- digital gift card",
        "ticket": "ORD-1008, I bought a gift card by accident. Please refund it.",
        "expected": "REJECTED",
    },
    {
        "id": "5a",
        "title": "Boundary -- $48.00, just under the $50 Standard cap",
        "ticket": (
            "Hi, my order ORD-1010 arrived damaged. It cost $48. Can I get "
            "a refund?"
        ),
        "expected": "AUTO_REFUND_APPROVED",
    },
    {
        "id": "5b",
        "title": "Boundary -- $52.00, just over the $50 Standard cap",
        "ticket": (
            "Hi, my order ORD-1011 arrived damaged. It cost $52. Can I get "
            "a refund?"
        ),
        "expected": "ESCALATION_REQUIRED",
    },
    {
        "id": "6",
        "title": "Risky customer -- repeat claims plus a fraud flag",
        "ticket": (
            "This is Ronen, order ORD-1005. The tablet screen was smashed "
            "on arrival. Refund me, this keeps happening."
        ),
        "expected": "ESCALATION_REQUIRED",
    },
    {
        "id": "7a",
        "title": "Order has not shipped -- still processing",
        "ticket": (
            "Hi, I'd like a refund for order ORD-1007, I don't want it "
            "anymore."
        ),
        "expected": "REJECTED",
    },
    {
        "id": "7b",
        "title": "Order has not shipped -- cancelled",
        "ticket": "Please refund order ORD-1009, I want my money back.",
        "expected": "REJECTED",
    },
    {
        "id": "9",
        "title": "Hallucination trap -- order does not exist",
        "ticket": "My order ORD-2222 never arrived and I want the $300 back.",
        "expected": "CANNOT_RESOLVE",
    },
]


def main() -> int:
    configure_logging()
    agent = ResolverAgent()
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
