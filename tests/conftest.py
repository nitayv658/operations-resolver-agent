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
