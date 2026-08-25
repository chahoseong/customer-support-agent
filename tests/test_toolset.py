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


def test_toolset_preserves_configured_name_when_returned_definition_name_changes() -> (
    None
):
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))

    returned_definition = toolset.definitions[0]
    returned_definition.name = "renamed_tool"

    original_name_result = toolset.execute(
        "configured_tool",
        {"value": "expected"},
        context=ToolContext(customer_id="customer-001"),
    )
    renamed_name_result = toolset.execute(
        "renamed_tool",
        {"value": "expected"},
        context=ToolContext(customer_id="customer-001"),
    )

    assert toolset.definitions[0].name == "configured_tool"
    assert original_name_result == "expected"
    assert renamed_name_result == {
        "error": {
            "code": "unknown_tool",
            "message": (
                "The requested tool is not available. Use an available tool instead."
            ),
        }
    }


def test_toolset_preserves_schema_when_returned_definition_parameters_change() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))
    returned_definition = toolset.definitions[0]

    returned_properties = returned_definition.parameters["properties"]
    assert isinstance(returned_properties, dict)

    returned_value_schema = returned_properties["value"]
    assert isinstance(returned_value_schema, dict)

    returned_value_schema["type"] = "integer"

    preserved_definition = toolset.definitions[0]

    preserved_properties = preserved_definition.parameters["properties"]
    assert isinstance(preserved_properties, dict)

    preserved_value_schema = preserved_properties["value"]
    assert isinstance(preserved_value_schema, dict)

    assert preserved_value_schema["type"] == "string"


def test_toolset_preserves_configuration_when_source_tool_definition_changes() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))

    configured_tool.definition.name = "renamed_tool"
    source_properties = configured_tool.definition.parameters["properties"]
    assert isinstance(source_properties, dict)

    source_value_schema = source_properties["value"]
    assert isinstance(source_value_schema, dict)

    source_value_schema["type"] = "integer"

    preserved_definition = toolset.definitions[0]

    assert preserved_definition.name == "configured_tool"

    preserved_properties = preserved_definition.parameters["properties"]
    assert isinstance(preserved_properties, dict)

    preserved_value_schema = preserved_properties["value"]
    assert isinstance(preserved_value_schema, dict)

    assert preserved_value_schema["type"] == "string"

    original_tool_result = toolset.execute(
        "configured_tool",
        {"value": "expected"},
        context=ToolContext(customer_id="customer-001"),
    )
    assert original_tool_result == "expected"

    renamed_tool_result = toolset.execute(
        "renamed_tool",
        {"value": "expected"},
        context=ToolContext(customer_id="customer-001"),
    )
    assert renamed_tool_result == {
        "error": {
            "code": "unknown_tool",
            "message": (
                "The requested tool is not available. Use an available tool instead."
            ),
        }
    }
