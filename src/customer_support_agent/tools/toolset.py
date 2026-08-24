from collections.abc import Sequence

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
        self._tools = tuple(tools)

        tool_names = [tool.definition.name for tool in self._tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Toolset tool names must be unique.")

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools)

    def execute(
        self,
        tool_name: str,
        arguments: object,
        *,
        context: ToolContext,
    ) -> object:
        for configured_tool in self._tools:
            if configured_tool.definition.name != tool_name:
                continue

            try:
                return configured_tool(
                    arguments,
                    context=context,
                )
            except Exception:
                return create_tool_error("tool_execution_failed")

        return create_tool_error("unknown_tool")
