"""CrewAI tool wrappers for Part 2's Researcher / Decision / Comms crew.

Generalizes resolver_agent_crewai.tools.build_tools() (hardcoded to Part 1's
4 GlobalCart tools plus cross-customer authorization -- a Part 1 concern
Part 2 doesn't have) into a build_tools(tool_names, tool_registry, call_log)
factory parameterized by whichever tool subset a role needs. Reuses
_args_model/_DispatchTool from that module directly -- they're generic
CrewAI-tool-building mechanics, not GlobalCart- or Part-1-specific.

Authority separation (which agent can reach which tool) lives here exactly
the way it does in the hand-rolled resolver_agent/crew/{researcher,decision,
comms}/agent.py: each build_*_tools() function only ever looks up names
from its own fixed tuple in multi_agent_tools.TOOL_REGISTRY, so e.g.
process_refund is physically unreachable from build_researcher_tools() or
build_comms_tools() regardless of what either agent is prompted to do.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from crewai.tools import BaseTool

from resolver_agent.tool_loop import ToolCallRecord

from ..tools import _DispatchTool, _args_model

STARTER_KIT_DIR = Path(__file__).resolve().parent.parent.parent / "starter-kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

import multi_agent_tools as mat  # noqa: E402  (path must be set up first)

from resolver_agent.crew.comms.agent import _guarded_registry  # noqa: E402

_RESEARCHER_TOOL_NAMES: Tuple[str, ...] = ("get_order_details", "get_user_profile", "audit_fraud_risk")
_DECISION_TOOL_NAMES: Tuple[str, ...] = ("check_return_policy", "process_refund")
_COMMS_TOOL_NAMES: Tuple[str, ...] = ("get_escalation_route", "send_slack_alert")

_SCHEMAS_BY_NAME: Dict[str, Dict[str, Any]] = {schema["name"]: schema for schema in mat.TOOL_SCHEMAS}


def build_tools(
    tool_names: Tuple[str, ...],
    tool_registry: Dict[str, Callable[..., Any]],
    call_log: List[ToolCallRecord],
) -> List[BaseTool]:
    """Build one fresh set of CrewAI tools for exactly ``tool_names``,
    tracing every call into ``call_log`` -- the same tracing role
    resolver_agent_crewai.tools.build_tools() plays for Part 1.
    """
    tools: List[BaseTool] = []
    for name in tool_names:
        schema = _SCHEMAS_BY_NAME[name]
        fn = tool_registry[name]

        def _tracked(fn: Callable[..., Any] = fn, name: str = name, **kwargs: Any) -> Any:
            result = fn(**kwargs)
            call_log.append(ToolCallRecord(name=name, input=dict(kwargs), result=result))
            return result

        tools.append(
            _DispatchTool(
                name=name,
                description=schema["description"],
                args_schema=_args_model(schema),
                dispatch=_tracked,
            )
        )
    return tools


def build_researcher_tools(call_log: List[ToolCallRecord]) -> List[BaseTool]:
    registry = {name: mat.TOOL_REGISTRY[name] for name in _RESEARCHER_TOOL_NAMES}
    return build_tools(_RESEARCHER_TOOL_NAMES, registry, call_log)


def build_decision_tools(call_log: List[ToolCallRecord]) -> List[BaseTool]:
    registry = {name: mat.TOOL_REGISTRY[name] for name in _DECISION_TOOL_NAMES}
    return build_tools(_DECISION_TOOL_NAMES, registry, call_log)


def build_comms_tools(call_log: List[ToolCallRecord], case_id: str) -> List[BaseTool]:
    base_registry = {name: mat.TOOL_REGISTRY[name] for name in _COMMS_TOOL_NAMES}
    guarded_registry = _guarded_registry(base_registry, case_id)
    return build_tools(_COMMS_TOOL_NAMES, guarded_registry, call_log)
