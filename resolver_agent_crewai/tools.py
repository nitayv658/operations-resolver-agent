"""CrewAI tool wrappers around the starter kit's GlobalCart tools.

CrewAI's `Agent.tools` is a fixed list built once per Agent, unlike
tool_loop.py's plain `Dict[str, Callable]` registry that ResolverAgent.resolve()
rebuilds fresh on every call (to apply per-call authorization -- see
resolver_agent.authorization). build_tools() reproduces that same per-call
freshness: CrewResolverAgent.resolve() calls it once per ticket, so each
ticket gets its own authorization-wrapped tools and its own call_log.

call_log stands in for tool_loop.py's automatic ToolCallRecord tracing --
CrewAI doesn't expose an equivalent trace of what a tool actually returned,
so each wrapper appends one here itself. Reusing the exact ToolCallRecord
type (not a plain dict) is what lets resolver_agent.output_tool.enforce_resolution
run against this list completely unmodified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr, create_model

from crewai.tools import BaseTool

STARTER_KIT_DIR = Path(__file__).resolve().parent.parent / "starter-kit"
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT_DIR))

import mock_services as gc  # noqa: E402  (path must be set up first)

from resolver_agent.authorization import authorize_tool_registry  # noqa: E402
from resolver_agent.tool_loop import ToolCallRecord  # noqa: E402

_JSON_TYPE_MAP = {"string": str, "number": float, "integer": int, "boolean": bool}


def _args_model(tool_schema: Dict[str, Any]) -> Type[BaseModel]:
    """Build a pydantic args model from one of mock_services.TOOL_SCHEMAS'
    JSON-schema `input_schema`, for CrewAI's `BaseTool.args_schema`.

    Reads the enum/required/type/description straight from the same schema
    Part 1 hands the Anthropic API -- so a tool's accepted arguments can
    never silently drift between the two implementations.
    """
    input_schema = tool_schema["input_schema"]
    properties: Dict[str, Any] = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    fields: Dict[str, Any] = {}
    for name, spec in properties.items():
        if "enum" in spec:
            py_type = Literal[tuple(spec["enum"])]  # type: ignore[valid-type]
        else:
            py_type = _JSON_TYPE_MAP.get(spec.get("type"), str)
        description = spec.get("description", "")
        if name in required:
            fields[name] = (py_type, Field(..., description=description))
        else:
            fields[name] = (Optional[py_type], Field(default=None, description=description))

    return create_model(f"{tool_schema['name']}_Args", **fields)  # type: ignore[call-overload]


class _DispatchTool(BaseTool):
    """A CrewAI BaseTool that just forwards to whatever callable it was
    built with -- the callable (set per-instance, per ticket) is what
    actually carries the authorization wrapping and call_log tracing.
    """

    _dispatch: Callable[..., Any] = PrivateAttr()

    def __init__(self, *, name: str, description: str, args_schema: Type[BaseModel], dispatch: Callable[..., Any]):
        super().__init__(name=name, description=description, args_schema=args_schema)
        self._dispatch = dispatch

    def _run(self, **kwargs: Any) -> Any:
        return self._dispatch(**kwargs)


def build_tools(
    call_log: List[ToolCallRecord],
    requester_user_id: Optional[str],
    log_context: Dict[str, Any],
) -> List[BaseTool]:
    """Build one fresh, ticket-scoped set of CrewAI tools.

    Applies the same cross-customer authorization substitution as
    resolver_agent.agent.ResolverAgent.resolve() (reusing
    authorize_tool_registry directly, unmodified), and appends a
    ToolCallRecord to ``call_log`` for every call so the CrewAI port can
    reuse resolver_agent.output_tool.validate_schema/enforce_resolution
    exactly as Part 1 does.
    """
    registry: Dict[str, Callable[..., Any]] = dict(gc.TOOL_REGISTRY)
    if requester_user_id is not None:
        registry = authorize_tool_registry(registry, requester_user_id, log_context)

    tools: List[BaseTool] = []
    for schema in gc.TOOL_SCHEMAS:
        name = schema["name"]
        fn = registry[name]

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
