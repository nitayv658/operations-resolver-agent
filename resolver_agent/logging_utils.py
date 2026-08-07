"""Structured logging helpers.

resolver_agent only ever emits through named loggers (``resolver_agent.*``)
-- it never calls ``logging.basicConfig()`` itself, and never attaches a
handler on import. Only an application entry point (``run_ticket.py``,
``run_scenarios.py``) calls :func:`configure_logging`. This is the normal
library/application split: a library that grabs the root logger on import
makes itself impossible to embed cleanly (e.g. inside Part 2's multi-agent
system, or a test suite using ``caplog``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit one structured log record.

    ``event`` is a short, dotted, greppable name (e.g. ``"tool_loop.max_iterations_reached"``),
    kept in the record's message. Everything else is a structured field,
    carried as ``record.fields`` rather than interpolated into the message
    string, so :class:`JsonFormatter` can render it without parsing text
    back out of a sentence.
    """
    logger.log(level, event, extra={"fields": fields})


class JsonFormatter(logging.Formatter):
    """Renders one JSON object per line: level, logger name, the event name
    (as ``event``), and every field passed to :func:`log_event` as its own
    top-level key.

    Deliberately does not go looking for anything beyond what the caller
    explicitly passed as a field -- no automatic dump of exception bodies,
    request payloads, etc., which is how PII ends up in logs by accident.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        return json.dumps(payload, default=str)


def configure_logging(level: Optional[str] = None) -> None:
    """Attach a single stderr handler with JSON output to the
    ``resolver_agent`` logger tree. Call this once, from a CLI entry point.

    Level resolution: the ``level`` argument, else the ``LOG_LEVEL`` env
    var, else ``WARNING`` -- quiet by default, so a normal successful run
    of ``run_ticket.py`` prints nothing to stderr.

    stderr, not stdout: ``run_ticket.py`` prints the agent's JSON result to
    stdout as its actual output contract. Logs on stdout would interleave
    with and corrupt that.
    """
    resolved = (level or os.environ.get("LOG_LEVEL") or "WARNING").upper()
    root = logging.getLogger("resolver_agent")
    root.setLevel(resolved)
    root.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False
