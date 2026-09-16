"""Skip this whole subpackage cleanly when CrewAI isn't importable.

resolver_agent_crewai/ needs CrewAI (Python >=3.10 -- see requirements-crewai.txt
/ .venv-crewai), which this repo's main .venv (3.9) never installs. Without
this guard, `pytest` from the main .venv would hard-fail collecting these
tests with a ModuleNotFoundError instead of skipping, breaking the existing
suite's green status for anyone who hasn't set up .venv-crewai.
"""

import pytest

pytest.importorskip(
    "crewai",
    reason="CrewAI not installed in this environment -- run these tests from .venv-crewai (see requirements-crewai.txt).",
)
