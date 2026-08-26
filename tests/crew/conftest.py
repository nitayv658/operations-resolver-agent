import sys
from pathlib import Path

STARTER_KIT_DIR = Path(__file__).resolve().parent.parent.parent / "starter-kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

import pytest  # noqa: E402

import multi_agent_tools as mat  # noqa: E402  (the real starter-kit module, not a mock)


@pytest.fixture(autouse=True)
def _isolate_outbox(tmp_path, monkeypatch):
    """multi_agent_tools.send_slack_alert's default outbox is a real
    starter-kit/outbox/alerts.jsonl file -- redirect it to a per-test
    tmp_path so running the suite never writes to (or races on) the real
    file. Same treatment tests/conftest.py already gives
    escalation_workflow.DEFAULT_QUEUE_PATH."""
    monkeypatch.setattr(mat, "OUTBOX_PATH", tmp_path / "alerts.jsonl")
    # send_slack_alert also reports outbox_path as OUTBOX_PATH.relative_to(
    # BASE_DIR) -- BASE_DIR must move with it or that call raises ValueError
    # (tmp_path is never a subpath of the real starter-kit dir). DATA_DIR is
    # already bound to the real data/ dir at import time and doesn't derive
    # from BASE_DIR at call time, so fixture loading is unaffected.
    monkeypatch.setattr(mat, "BASE_DIR", tmp_path)
    # A real SLACK_WEBHOOK_URL in the developer's shell must not make the
    # suite attempt real network calls -- tests that want the webhook path
    # opt in explicitly via monkeypatch.setenv.
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)


@pytest.fixture
def mat_module():
    """The real multi_agent_tools module -- for tests that need to call a
    tool directly (e.g. to compute an expected value) rather than only
    through a scripted agent run."""
    return mat
