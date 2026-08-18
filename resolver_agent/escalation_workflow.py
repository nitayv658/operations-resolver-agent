"""Stage-9 "trigger workflow" side effect for a resolved case.

submit_resolution already covers "respond" (customer_response) and "write" (the
returned dict, logged via agent.case_resolved / agent.resolution_corrected).
What was missing: when a case resolves to something a human still has to act
on (ESCALATION_REQUIRED, CANNOT_RESOLVE), nothing created an artifact that
human could actually pick up -- customer_response saying "we've escalated it"
was only ever a promise inside the LLM's own reply text, with no downstream
effect.

Deliberately minimal, consistent with this repo's own stated scope
boundaries (see resolver_agent/agent.py, README "Edge cases and guardrails"):
no external ticketing system, no queue infrastructure, and this module never
reads its own output back -- resolve() stays stateless across calls, this is
a one-way write for an external ops process to consume. A real deployment
would swap _append_jsonl for an actual API call (PagerDuty, Zendesk, a
ticket-creation endpoint); the injectable ``writer`` exists for exactly that.

Raw ticket text and customer_response are never included in the record, for
the same privacy reason logging_utils.py never logs them (can contain a
customer's name) -- only structural facts, the same shape already exposed in
reasoning_chain/action_taken.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# The two decision values that mean "a human still needs to act" -- see
# output_tool.DECISION_VALUES for the full set. AUTO_REFUND_APPROVED and
# REJECTED are terminal: nothing downstream needs to happen.
DECISIONS_NEEDING_WORKFLOW = frozenset({"ESCALATION_REQUIRED", "CANNOT_RESOLVE"})

# Read once at import time via env var override, same pattern as
# agent.DEFAULT_MODEL. Referenced through the module attribute (not bound as
# a default parameter value) so tests can monkeypatch it per-test.
DEFAULT_QUEUE_PATH = Path(os.environ.get("ESCALATION_QUEUE_PATH", "escalation_queue.jsonl"))


def build_escalation_record(resolution: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    """The structural-only record written for a case a human must act on."""
    action = resolution.get("action_taken") or {}
    return {
        "case_id": case_id,
        "decision": action.get("decision"),
        "tools_called": action.get("tools_called", []),
        "reasoning_chain": resolution.get("reasoning_chain", []),
        "corrections": resolution.get("_corrections", []),
    }


def _append_jsonl(record: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def trigger_workflow(
    resolution: Dict[str, Any],
    case_id: str,
    *,
    queue_path: Optional[Path] = None,
    writer: Optional[Callable[[Dict[str, Any], Path], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Write an ops-queue record if, and only if, ``resolution`` needs one.

    Returns the record that was written, or ``None`` if this decision is
    terminal and no downstream action is needed (the common case -- most
    resolutions are AUTO_REFUND_APPROVED or REJECTED and this is a no-op).

    ``queue_path``/``writer`` are injection points for callers/tests --
    omitting both appends to :data:`DEFAULT_QUEUE_PATH` via
    :func:`_append_jsonl`, the real default behavior.
    """
    action = resolution.get("action_taken") or {}
    decision = action.get("decision")
    if decision not in DECISIONS_NEEDING_WORKFLOW:
        return None

    record = build_escalation_record(resolution, case_id)
    path = queue_path if queue_path is not None else DEFAULT_QUEUE_PATH
    write = writer if writer is not None else _append_jsonl
    write(record, path)
    return record
