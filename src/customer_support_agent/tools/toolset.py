from collections.abc import Sequence
from copy import deepcopy

from .errors import create_tool_error
from .tool import (
    Tool,
    ToolContext,
    ToolDefinition,
)


class Toolset:
    def __init__(
        self,
        *,
        tools: Sequence[Tool[object]],
    ) -> None:
        configured_tools = tuple(tools)

        self._definitions = deepcopy(
            tuple(configured_tool.definition for configured_tool in configured_tools)
        )
        self._tools_by_name = {
            configured_tool.definition.name: configured_tool
            for configured_tool in configured_tools
        }
        tool_names = [tool.definition.name for tool in configured_tools]
        if len(tool_names) != len(self._tools_by_name):
            raise ValueError("Toolset tool names must be unique.")

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return deepcopy(self._definitions)

    def execute(
        self,
        tool_name: str,
        arguments: object,
        *,
        context: ToolContext,
    ) -> object:
        configured_tool = self._tools_by_name.get(tool_name)
        if configured_tool is None:
            return create_tool_error("unknown_tool")

        try:
            return configured_tool(
                arguments,
                context=context,
            )
        except Exception:
            return create_tool_error("tool_execution_failed")
