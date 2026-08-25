from collections.abc import Callable
from typing import Annotated

import pytest
from pydantic import Field

from customer_support_agent.tools.tool import ToolContext, tool

DEFAULT_TOOL_CONTEXT = ToolContext(customer_id="customer-001")


def executor_with_positional_only_parameter(
    value: str,
    /,
) -> str:
    """Return the provided example value."""
    return value


def executor_with_variadic_positional_parameter(
    *values: str,
) -> str:
    """Return the first provided example value."""
    return values[0]


def executor_with_variadic_keyword_parameter(
    **values: str,
) -> str:
    """Return the first provided example value."""
    return next(iter(values.values()))


def test_tool_exposes_definition_from_executor_signature() -> None:
    @tool
    def example_tool(value: str) -> str:
        """Return the provided example value."""
        return value

    assert example_tool.definition.name == "example_tool"
    assert example_tool.definition.description == "Return the provided example value."

    parameters = example_tool.definition.parameters
    assert parameters["type"] == "object"
    assert parameters["required"] == ["value"]
    assert parameters["additionalProperties"] is False

    properties = parameters["properties"]
    assert isinstance(properties, dict)

    value_schema = properties["value"]
    assert isinstance(value_schema, dict)
    assert value_schema["type"] == "string"


def test_tool_definition_preserves_parameter_annotation_metadata() -> None:
    @tool
    def example_tool(
        value: Annotated[
            str,
            Field(
                min_length=2,
                description="Example value to return.",
            ),
        ],
    ) -> str:
        """Return the provided example value."""
        return value

    properties = example_tool.definition.parameters["properties"]
    assert isinstance(properties, dict)

    value_schema = properties["value"]
    assert isinstance(value_schema, dict)
    assert value_schema["minLength"] == 2
    assert value_schema["description"] == "Example value to return."


def test_tool_raises_type_error_when_tool_context_follows_exposed_parameter() -> None:
    def example_tool(value: str, context: ToolContext) -> str:
        """Return the provided example value."""
        return f"{context.customer_id}: {value}"

    with pytest.raises(TypeError):
        tool(example_tool)


def test_tool_raises_type_error_when_tool_context_is_keyword_only() -> None:
    def example_tool(*, context: ToolContext) -> str:
        """Return the current customer."""
        return context.customer_id

    with pytest.raises(TypeError):
        tool(example_tool)


@pytest.mark.parametrize(
    "docstring",
    [
        pytest.param(None, id="missing"),
        pytest.param("   ", id="blank"),
    ],
)
def test_tool_raises_value_error_when_executor_docstring_is_missing_or_blank(
    docstring: str | None,
) -> None:
    def example_tool(value: str) -> str:
        return value

    example_tool.__doc__ = docstring

    with pytest.raises(ValueError):
        tool(example_tool)


def test_tool_raises_type_error_without_executor_parameter_annotation() -> None:
    def example_tool(value: str) -> str:
        """Return the provided example value."""
        return value

    del example_tool.__annotations__["value"]

    with pytest.raises(TypeError):
        tool(example_tool)


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(
            executor_with_positional_only_parameter,
            id="positional-only",
        ),
        pytest.param(
            executor_with_variadic_positional_parameter,
            id="variadic-positional",
        ),
        pytest.param(
            executor_with_variadic_keyword_parameter,
            id="variadic-keyword",
        ),
    ],
)
def test_tool_raises_type_error_for_unsupported_executor_parameter_kind(
    executor: Callable[..., str],
) -> None:
    with pytest.raises(TypeError):
        tool(executor)


def test_tool_definition_marks_parameter_with_default_as_optional() -> None:
    @tool
    def example_tool(value: str, locale: str = "ko") -> str:
        """Return the provided example value."""
        return f"{locale}: {value}"

    parameters = example_tool.definition.parameters
    assert parameters["required"] == ["value"]

    properties = parameters["properties"]
    assert isinstance(properties, dict)

    locale_schema = properties["locale"]
    assert isinstance(locale_schema, dict)
    assert locale_schema["default"] == "ko"


def test_tool_raises_type_error_for_tool_context_parameter_with_default() -> None:
    def example_tool(
        context: ToolContext = DEFAULT_TOOL_CONTEXT,
    ) -> str:
        """Return the current customer."""
        return context.customer_id

    with pytest.raises(TypeError):
        tool(example_tool)
