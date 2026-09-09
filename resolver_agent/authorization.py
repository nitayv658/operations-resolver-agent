"""Cross-customer authorization: deny a tool result that names a different
owner than the requester, before it ever reaches the model.

Split out of agent.py (which originally inlined this) for the same reason
escalation_workflow.py already lives in its own file: agent.py's own job is
wiring GlobalCart's tools and submit_resolution into tool_loop, not owning
every concern that touches those tools. This module is a self-contained,
independently testable piece: given a tool_registry and a requester
identity, it returns an authorization-wrapped registry.

This is prevention, not detection -- the substitution happens at the
tool-dispatch boundary, so the real data never enters the model's context in
the first place, rather than being caught only in the final customer-facing
output. See ResolverAgent.resolve()'s docstring (agent.py) for the full
requester_user_id contract this supports.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from .logging_utils import get_logger, log_event

_logger = get_logger(__name__)

# Every GlobalCart tool's successful result carries this field naming the
# owning customer (confirmed by reading mock_services.py: get_order_details,
# get_user_profile, check_return_policy and process_refund all include it).
OWNER_FIELD = "user_id"


def deny_unauthorized() -> Dict[str, Any]:
    return {
        "error": "NOT_AUTHORIZED",
        "message": "This record does not belong to the requesting customer.",
    }


def authorize_tool_registry(
    tool_registry: Dict[str, Callable[..., Any]],
    requester_user_id: str,
    log_context: Dict[str, Any],
) -> Dict[str, Callable[..., Any]]:
    """Wrap every tool so a result belonging to a different customer than
    ``requester_user_id`` is replaced with :func:`deny_unauthorized` before
    it ever reaches the model.

    The NOT_AUTHORIZED shape matches every other business failure in this
    codebase (an ``error`` key), so it needs no special handling anywhere
    else: prompts.py's existing "if a tool result contains an error key"
    rule and output_tool.py's existing "a tool errored, no refund was
    processed" enforcement rule both already cover it generically, with zero
    changes to either file.

    Genuine tool errors (e.g. ORDER_NOT_FOUND) are untouched -- only a
    successful result naming a *different* owner gets substituted.
    """

    def _wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(**kwargs: Any) -> Any:
            result = fn(**kwargs)
            if isinstance(result, dict) and "error" not in result:
                owner = result.get(OWNER_FIELD)
                if owner is not None and owner != requester_user_id:
                    log_event(
                        _logger,
                        logging.WARNING,
                        "agent.unauthorized_tool_result_denied",
                        tool=name,
                        record_owner=owner,
                        requester_user_id=requester_user_id,
                        **log_context,
                    )
                    return deny_unauthorized()
            return result

        return wrapped

    return {name: _wrap(name, fn) for name, fn in tool_registry.items()}
