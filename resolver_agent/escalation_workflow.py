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
a one-way write for an external ops process to consume.

Two delivery mechanisms exist:

- ``_append_jsonl`` -- the local-file default, always available, no
  configuration needed.
- ``build_webhook_writer`` -- POSTs the record as JSON to any URL, which
  covers most real ops systems without picking a specific vendor SDK
  (Zendesk triggers, PagerDuty's Events API, Opsgenie, Slack incoming
  webhooks, and a bespoke internal endpoint are all "POST me some JSON").
  Set ``ESCALATION_WEBHOOK_URL`` and :func:`trigger_workflow`'s default
  writer switches to it automatically. A failed delivery (network error,
  timeout, non-2xx) never loses the record -- it falls back to the same
  local JSONL append the file-only default would have used, so an ops
  endpoint being briefly down can't make an escalation vanish silently.

Raw ticket text and customer_response are never included in the record, for
the same privacy reason logging_utils.py never logs them (can contain a
customer's name) -- only structural facts, the same shape already exposed in
reasoning_chain/action_taken.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .logging_utils import get_logger, log_event

_logger = get_logger(__name__)

# The two decision values that mean "a human still needs to act" -- see
# output_tool.DECISION_VALUES for the full set. AUTO_REFUND_APPROVED and
# REJECTED are terminal: nothing downstream needs to happen.
DECISIONS_NEEDING_WORKFLOW = frozenset({"ESCALATION_REQUIRED", "CANNOT_RESOLVE"})

# Read once at import time via env var override, same pattern as
# agent.DEFAULT_MODEL. Referenced through the module attribute (not bound as
# a default parameter value) so tests can monkeypatch it per-test.
DEFAULT_QUEUE_PATH = Path(os.environ.get("ESCALATION_QUEUE_PATH", "escalation_queue.jsonl"))

DEFAULT_WEBHOOK_TIMEOUT_SECONDS = 5.0

# Indirection so tests can monkeypatch the actual network call without
# touching the real internet -- same DI shape as everything else in this
# module (queue_path, writer).
_urlopen = urllib.request.urlopen


class WebhookDeliveryError(RuntimeError):
    """A webhook POST failed (network error, timeout, or non-2xx response).

    Caught internally by the writer :func:`build_webhook_writer` returns --
    callers never see this directly, they see a fallback file write and a
    logged warning instead. A delivery failure is an infra hiccup, not a
    reason to lose an escalation record or crash resolve().
    """


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


def post_webhook(record: Dict[str, Any], url: str, *, timeout: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS) -> None:
    """POST ``record`` as JSON to ``url``. Raises :class:`WebhookDeliveryError`
    on any network error, timeout, or non-2xx response -- ``urlopen`` itself
    already raises ``HTTPError`` (a subclass of ``URLError``) for a non-2xx
    status, so a single except clause covers both.
    """
    data = json.dumps(record, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with _urlopen(request, timeout=timeout):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise WebhookDeliveryError(f"POST {url} failed: {exc}") from exc


def build_webhook_writer(
    url: str, *, timeout: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS
) -> Callable[[Dict[str, Any], Path], None]:
    """A writer that POSTs the record to ``url``, falling back to the local
    JSONL append (the same ``path`` :func:`trigger_workflow` would have used
    anyway) if delivery fails. A failed webhook is an infra hiccup, not a
    reason to lose the record or crash the case's resolve() call -- see
    :class:`WebhookDeliveryError`.
    """

    def _write(record: Dict[str, Any], path: Path) -> None:
        try:
            post_webhook(record, url, timeout=timeout)
        except WebhookDeliveryError as exc:
            log_event(
                _logger,
                logging.WARNING,
                "escalation_workflow.webhook_delivery_failed",
                url=url,
                error=str(exc),
                case_id=record.get("case_id"),
            )
            _append_jsonl(record, path)

    return _write


def _default_writer() -> Callable[[Dict[str, Any], Path], None]:
    """No explicit ``writer`` was passed to :func:`trigger_workflow` -- use
    the webhook if ``ESCALATION_WEBHOOK_URL`` is configured, otherwise the
    local file. Read from the environment at call time (not import time)
    so tests can toggle it per-test without reloading the module.
    """
    url = os.environ.get("ESCALATION_WEBHOOK_URL")
    if url:
        return build_webhook_writer(url)
    return _append_jsonl


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

    ``queue_path``/``writer`` are injection points for callers/tests.
    Omitting ``writer`` uses :func:`_default_writer` -- the webhook if
    ``ESCALATION_WEBHOOK_URL`` is set, otherwise :func:`_append_jsonl`.
    Omitting ``queue_path`` uses :data:`DEFAULT_QUEUE_PATH`, which still
    matters even with the webhook writer active: it's the fallback path
    used if a webhook delivery fails.
    """
    action = resolution.get("action_taken") or {}
    decision = action.get("decision")
    if decision not in DECISIONS_NEEDING_WORKFLOW:
        return None

    record = build_escalation_record(resolution, case_id)
    path = queue_path if queue_path is not None else DEFAULT_QUEUE_PATH
    write = writer if writer is not None else _default_writer()
    write(record, path)
    return record
