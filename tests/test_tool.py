from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel, ConfigDict

from customer_support_agent.tools.tool import ToolContext, ToolDefinition, tool


class ExampleArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    value: str


DEFAULT_EXAMPLE_ARGUMENTS = ExampleArguments(value="default")


def executor_with_positional_only_parameter(
    arguments: ExampleArguments,
    /,
) -> str:
    """Return the provided example value."""
    return arguments.value


def executor_with_keyword_only_parameter(
    *,
    arguments: ExampleArguments,
) -> str:
    """Return the provided example value."""
    return arguments.value


def executor_with_variadic_positional_parameter(
    *arguments: ExampleArguments,
) -> str:
    """Return the first provided example value."""
    return arguments[0].value


def executor_with_variadic_keyword_parameter(
    **arguments: ExampleArguments,
) -> str:
    """Return the first provided example value."""
    return next(iter(arguments.values())).value


def test_tool_exposes_definition_from_executor() -> None:
    @tool
    def example_tool(arguments: ExampleArguments) -> str:
        """Return the provided example value."""
        return arguments.value

    assert example_tool.definition == ToolDefinition(
        name="example_tool",
        description="Return the provided example value.",
        parameters=ExampleArguments.model_json_schema(),
    )


def test_tool_passes_validated_arguments_to_executor() -> None:
    received_arguments: list[ExampleArguments] = []

    @tool
    def example_tool(arguments: ExampleArguments) -> str:
        """Return the provided example value."""
        received_arguments.append(arguments)
        return arguments.value

    result = example_tool({"value": "expected"})

    assert result == "expected"
    assert received_arguments == [ExampleArguments(value="expected")]


def test_tool_returns_invalid_arguments_when_arguments_do_not_match_schema() -> None:
    @tool
    def example_tool(arguments: ExampleArguments) -> str:
        """Return the provided example value."""
        return arguments.value

    result = example_tool({"value": 123})

    assert result == {
        "error": {
            "code": "invalid_arguments",
            "message": "Arguments do not match the tool's input schema.",
        }
    }


def test_tool_passes_tool_context_to_executor() -> None:
    received_contexts: list[ToolContext] = []

    @tool
    def example_tool(
        context: ToolContext,
        arguments: ExampleArguments,
    ) -> str:
        """Return the provided example value."""
        received_contexts.append(context)
        return arguments.value

    context = ToolContext(customer_id="customer-001")

    result = example_tool(
        {"value": "expected"},
        context=context,
    )

    assert result == "expected"
    assert received_contexts == [context]


def test_tool_raises_type_error_without_required_tool_context() -> None:
    @tool
    def example_tool(
        context: ToolContext,
        arguments: ExampleArguments,
    ) -> str:
        """Return the provided example value."""
        return arguments.value

    with pytest.raises(TypeError):
        example_tool({"value": "expected"})


def test_tool_context_rejects_customer_id_assignment() -> None:
    context = ToolContext(customer_id="customer-001")

    with pytest.raises(FrozenInstanceError):
        context.customer_id = "customer-002"  # type: ignore[misc]


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
    def example_tool(arguments: ExampleArguments) -> str:
        return arguments.value

    example_tool.__doc__ = docstring

    with pytest.raises(ValueError):
        tool(example_tool)


def test_tool_raises_type_error_without_arguments_parameter() -> None:
    def example_tool() -> str:
        """Return an example value."""
        return "example"

    with pytest.raises(TypeError):
        tool(example_tool)


def test_tool_raises_type_error_without_arguments_annotation() -> None:
    def example_tool(arguments: ExampleArguments) -> str:
        """Return the provided example value."""
        return arguments.value

    del example_tool.__annotations__["arguments"]

    with pytest.raises(TypeError):
        tool(example_tool)


@pytest.mark.parametrize(
    "arguments_type",
    [
        pytest.param(str, id="non-base-model"),
        pytest.param(BaseModel, id="base-model-itself"),
    ],
)
def test_tool_raises_type_error_without_concrete_arguments_model(
    arguments_type: object,
) -> None:
    def example_tool(arguments: ExampleArguments) -> str:
        """Return the provided example value."""
        return arguments.value

    example_tool.__annotations__["arguments"] = arguments_type

    with pytest.raises(TypeError):
        tool(example_tool)


def test_tool_raises_type_error_for_two_parameter_executor_without_tool_context() -> (
    None
):
    def example_tool(
        customer_id: str,
        arguments: ExampleArguments,
    ) -> str:
        """Return the provided example value."""
        return f"{customer_id}: {arguments.value}"

    with pytest.raises(TypeError):
        tool(example_tool)


def test_tool_raises_type_error_with_more_than_two_executor_parameters() -> None:
    def example_tool(
        context: ToolContext,
        locale: str,
        arguments: ExampleArguments,
    ) -> str:
        """Return the provided example value."""
        return f"{context.customer_id}: {locale}: {arguments.value}"

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
            executor_with_keyword_only_parameter,
            id="keyword-only",
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


def test_tool_raises_type_error_for_executor_parameter_with_default() -> None:
    def example_tool(
        arguments: ExampleArguments = DEFAULT_EXAMPLE_ARGUMENTS,
    ) -> str:
        """Return the provided example value."""
        return arguments.value

    with pytest.raises(TypeError):
        tool(example_tool)


def test_tool_passes_only_arguments_to_executor_without_tool_context_parameter() -> (
    None
):
    @tool
    def example_tool(arguments: ExampleArguments) -> str:
        """Return the provided example value."""
        return arguments.value

    result = example_tool(
        {"value": "expected"},
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == "expected"


def test_tool_raises_type_error_for_missing_tool_context_before_validating_arguments() -> (
    None
):
    @tool
    def example_tool(
        context: ToolContext,
        arguments: ExampleArguments,
    ) -> str:
        """Return the provided example value."""
        return arguments.value

    with pytest.raises(TypeError):
        example_tool({"value": 123})
