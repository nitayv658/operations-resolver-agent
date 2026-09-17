"""Same outbox isolation as tests/crew/conftest.py -- multi_agent_tools's
default outbox is a real starter-kit/outbox/alerts.jsonl file; redirect it
to a per-test tmp_path so this suite never writes to (or races on) the real
file. pytest.importorskip("crewai") is inherited from tests/crewai/conftest.py
(conftest guards cascade to subdirectories) -- not duplicated here.
"""

import sys
from pathlib import Path

STARTER_KIT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "starter-kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

import pytest  # noqa: E402

import multi_agent_tools as mat  # noqa: E402  (the real starter-kit module, not a mock)


@pytest.fixture(autouse=True)
def _isolate_outbox(tmp_path, monkeypatch):
    monkeypatch.setattr(mat, "OUTBOX_PATH", tmp_path / "alerts.jsonl")
    monkeypatch.setattr(mat, "BASE_DIR", tmp_path)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
