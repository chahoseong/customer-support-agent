import pytest

from customer_support_agent.tools.tool import Tool, ToolContext, tool
from customer_support_agent.tools.toolset import Toolset


def test_toolset_exposes_definitions_in_configured_order() -> None:
    @tool
    def first_tool(value: str) -> str:
        """Return the first value."""
        return value

    @tool
    def second_tool(value: str) -> str:
        """Return the second value."""
        return value

    toolset = Toolset(
        tools=(
            second_tool,
            first_tool,
        )
    )

    assert toolset.definitions == (
        second_tool.definition,
        first_tool.definition,
    )


def test_toolset_raises_value_error_for_duplicate_tool_names() -> None:
    def create_duplicate_tool(result: str) -> Tool[str]:
        @tool
        def duplicate_tool(value: str) -> str:
            """Return a configured value."""
            return f"{result}: {value}"

        return duplicate_tool

    first_tool = create_duplicate_tool("first")
    second_tool = create_duplicate_tool("second")

    with pytest.raises(ValueError):
        Toolset(
            tools=(
                first_tool,
                second_tool,
            )
        )


def test_toolset_executes_configured_tool_by_name() -> None:
    @tool
    def unselected_tool(value: str) -> str:
        """Return an unselected value."""
        return f"unselected: {value}"

    @tool
    def selected_tool(
        context: ToolContext,
        value: str,
    ) -> str:
        """Return the selected customer value."""
        return f"{context.customer_id}: {value}"

    toolset = Toolset(
        tools=(
            unselected_tool,
            selected_tool,
        )
    )

    result = toolset.execute(
        "selected_tool",
        {"value": "expected"},
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == "customer-001: expected"


def test_toolset_returns_unknown_tool_when_name_is_not_configured() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))

    result = toolset.execute(
        "unknown_tool",
        {"value": "expected"},
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == {
        "error": {
            "code": "unknown_tool",
            "message": (
                "The requested tool is not available. Use an available tool instead."
            ),
        }
    }


def test_toolset_returns_tool_execution_failed_when_executor_raises() -> None:
    @tool
    def failing_tool(value: str) -> str:
        """Fail while processing the provided value."""
        raise RuntimeError(f"sensitive failure: {value}")

    toolset = Toolset(tools=(failing_tool,))

    result = toolset.execute(
        "failing_tool",
        {"value": "secret"},
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == {
        "error": {
            "code": "tool_execution_failed",
            "message": "The tool failed unexpectedly; do not assume a result.",
        }
    }


def test_toolset_preserves_configuration_when_source_collection_changes() -> None:
    @tool
    def first_tool(value: str) -> str:
        """Return the first value."""
        return value

    @tool
    def second_tool(value: str) -> str:
        """Return the second value."""
        return value

    source_tools = [first_tool]
    toolset = Toolset(tools=source_tools)

    source_tools.append(second_tool)

    assert toolset.definitions == (first_tool.definition,)


def test_toolset_returns_invalid_arguments_when_arguments_do_not_match_tool_schema() -> (
    None
):
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))

    result = toolset.execute(
        "configured_tool",
        {"value": 123},
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == {
        "error": {
            "code": "invalid_arguments",
            "message": "Arguments do not match the tool's input schema.",
        }
    }
