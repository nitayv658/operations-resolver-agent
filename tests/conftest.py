import logging
import sys
from pathlib import Path

# Make `resolver_agent` importable regardless of how pytest is invoked --
# pytest only auto-adds each test file's own directory (tests/), not the
# project root, when tests/ has no __init__.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from resolver_agent.agent import gc  # noqa: E402  (the real starter-kit mock_services module)
from resolver_agent.output_tool import SUBMIT_RESOLUTION_SCHEMA  # noqa: E402


@pytest.fixture
def tool_schemas():
    """The exact tool list ResolverAgent hands to the model: the 4 real
    GlobalCart tools plus submit_resolution."""
    return list(gc.TOOL_SCHEMAS) + [SUBMIT_RESOLUTION_SCHEMA]


@pytest.fixture
def tool_registry():
    """The exact registry run_tool_loop dispatches real tool_use calls
    through -- the real starter-kit functions, not a mock of them."""
    return dict(gc.TOOL_REGISTRY)


@pytest.fixture(autouse=True)
def _isolate_escalation_queue(tmp_path, monkeypatch):
    """escalation_workflow's default queue path is a real repo-root file --
    redirect it to a per-test tmp_path so running the suite never writes to
    (or races on) the real escalation_queue.jsonl. Patched on the module
    attribute, not a bound default, since trigger_workflow reads it at call
    time -- see escalation_workflow.DEFAULT_QUEUE_PATH.
    """
    from resolver_agent import escalation_workflow

    monkeypatch.setattr(escalation_workflow, "DEFAULT_QUEUE_PATH", tmp_path / "escalation_queue.jsonl")
    # A real ESCALATION_WEBHOOK_URL in the developer's shell must not make
    # the suite attempt real network calls -- tests that want the webhook
    # path opt in explicitly via monkeypatch.setenv.
    monkeypatch.delenv("ESCALATION_WEBHOOK_URL", raising=False)


@pytest.fixture(autouse=True)
def _reset_resolver_agent_logging():
    """configure_logging() mutates global state on the "resolver_agent"
    logger (handlers, level, and propagate=False so a real application
    doesn't double-log to both our handler and the root's). That mutation
    leaking from one test into the next test file is exactly the kind of
    global-state bug it introduces -- propagate=False in particular blocks
    pytest's caplog (which listens at the root logger) from seeing records
    emitted by any test that runs after a test calling configure_logging().
    Capture and restore around every test so ordering can't matter.
    """
    logger = logging.getLogger("resolver_agent")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    yield
    logger.handlers[:] = original_handlers
    logger.setLevel(original_level)
    logger.propagate = original_propagate
