from collections.abc import Sequence
from copy import deepcopy

import logfire

from .errors import create_tool_error, get_tool_error_code
from .tool import (
    Tool,
    ToolContext,
    ToolDefinition,
)


def _get_tool_argument_names(arguments: object) -> list[str]:
    if type(arguments) is not dict:
        return []

    return sorted(key for key in arguments if type(key) is str)


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
        with logfire.span("tool.execute") as tool_span:
            tool_span.set_attributes(
                {
                    "customer_support_agent.tool.name": tool_name,
                    "customer_support_agent.tool.argument.names": (
                        _get_tool_argument_names(arguments)
                    ),
                }
            )

            configured_tool = self._tools_by_name.get(tool_name)
            result: object

            if configured_tool is None:
                result = create_tool_error("unknown_tool")
            else:
                try:
                    result = configured_tool(
                        arguments,
                        context=context,
                    )
                except Exception as error:
                    tool_span.set_attributes(
                        {
                            "customer_support_agent.tool.outcome": "exception",
                            "customer_support_agent.tool.error.code": (
                                "tool_execution_failed"
                            ),
                            "error.type": type(error).__name__,
                        }
                    )
                    tool_span.set_level("error")
                    return create_tool_error("tool_execution_failed")

            tool_error_code = get_tool_error_code(result)

            if tool_error_code is None:
                tool_span.set_attributes(
                    {
                        "customer_support_agent.tool.outcome": "success",
                    }
                )
            else:
                tool_span.set_attributes(
                    {
                        "customer_support_agent.tool.outcome": "tool_error",
                        "customer_support_agent.tool.error.code": tool_error_code,
                    }
                )

            return result
