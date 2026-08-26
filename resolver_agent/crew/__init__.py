"""The Part 2 multi-agent crew: Researcher -> Decision -> Comms."""

from __future__ import annotations

# resolver_agent.agent already puts starter-kit/ on sys.path (idempotent --
# guarded by `if ... not in sys.path`) so `import mock_services` resolves.
# Importing it here, before any crew submodule does `import multi_agent_tools`,
# reuses that same setup instead of duplicating the STARTER_KIT_DIR dance.
from .. import agent as _part1_agent  # noqa: F401

from .orchestrator import OperationsCrew

__all__ = ["OperationsCrew"]
