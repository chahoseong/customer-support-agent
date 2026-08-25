from collections.abc import Callable
from dataclasses import FrozenInstanceError
from enum import StrEnum
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

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


def test_tool_passes_validated_arguments_to_executor() -> None:
    received_values: list[str] = []

    @tool
    def example_tool(value: str) -> str:
        """Return the provided example value."""
        received_values.append(value)
        return value

    result = example_tool({"value": "expected"})

    assert result == "expected"
    assert received_values == ["expected"]


def test_tool_passes_validated_arguments_as_declared_python_types() -> None:
    class ExampleStatus(StrEnum):
        READY = "ready"

    class ExampleRequest(BaseModel):
        status: ExampleStatus

    received_requests: list[ExampleRequest] = []

    @tool
    def example_tool(request: ExampleRequest) -> str:
        """Return the provided example status."""
        received_requests.append(request)
        return request.status.value

    result = example_tool({"request": {"status": "ready"}})

    assert result == "ready"
    assert len(received_requests) == 1
    assert isinstance(received_requests[0], ExampleRequest)
    assert received_requests[0].status is ExampleStatus.READY


def test_tool_returns_invalid_arguments_when_arguments_do_not_match_schema() -> None:
    @tool
    def example_tool(value: str) -> str:
        """Return the provided example value."""
        return value

    result = example_tool({"value": 123})

    assert result == {
        "error": {
            "code": "invalid_arguments",
            "message": "Arguments do not match the tool's input schema.",
        }
    }


def test_tool_returns_invalid_arguments_when_parameter_annotation_constraint_fails() -> (
    None
):
    @tool
    def example_tool(value: Annotated[str, Field(min_length=2)]) -> str:
        """Return the provided example value."""
        return value

    result = example_tool({"value": "x"})

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
        value: str,
    ) -> str:
        """Return the provided example value."""
        received_contexts.append(context)
        return value

    context = ToolContext(customer_id="customer-001")

    result = example_tool(
        {"value": "expected"},
        context=context,
    )

    assert result == "expected"
    assert received_contexts == [context]


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


def test_tool_raises_type_error_without_required_tool_context() -> None:
    @tool
    def example_tool(
        context: ToolContext,
        value: str,
    ) -> str:
        """Return the provided example value."""
        return value

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
    def example_tool(value: str) -> str:
        return value

    example_tool.__doc__ = docstring

    with pytest.raises(ValueError):
        tool(example_tool)


def test_tool_returns_result_when_executor_declares_no_parameters() -> None:
    @tool
    def example_tool() -> str:
        """Return an example value."""
        return "example"

    result = example_tool({})

    assert result == "example"


def test_tool_raises_type_error_without_executor_parameter_annotation() -> None:
    def example_tool(value: str) -> str:
        """Return the provided example value."""
        return value

    del example_tool.__annotations__["value"]

    with pytest.raises(TypeError):
        tool(example_tool)


def test_tool_passes_multiple_validated_arguments_to_executor() -> None:
    received_arguments: list[tuple[str, int]] = []

    @tool
    def example_tool(
        value: str,
        count: int,
    ) -> str:
        """Repeat the provided example value."""
        received_arguments.append((value, count))
        return value * count

    result = example_tool(
        {
            "value": "expected",
            "count": 2,
        }
    )

    assert result == "expectedexpected"
    assert received_arguments == [("expected", 2)]


def test_tool_passes_multiple_validated_arguments_with_tool_context() -> None:
    @tool
    def example_tool(
        context: ToolContext,
        value: str,
        count: int,
    ) -> str:
        """Repeat the provided example value for the customer."""
        return f"{context.customer_id}: {value * count}"

    context = ToolContext(customer_id="customer-001")

    result = example_tool(
        {
            "value": "expected",
            "count": 2,
        },
        context=context,
    )

    assert result == "customer-001: expectedexpected"


def test_tool_passes_validated_argument_to_keyword_only_parameter() -> None:
    @tool
    def example_tool(*, value: str) -> str:
        """Return the provided example value."""
        return value

    result = example_tool({"value": "expected"})

    assert result == "expected"


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


def test_tool_passes_parameter_default_when_argument_is_omitted() -> None:
    @tool
    def example_tool(value: str, locale: str = "ko") -> str:
        """Return the provided example value."""
        return f"{locale}: {value}"

    result = example_tool({"value": "expected"})

    assert result == "ko: expected"


def test_tool_raises_type_error_for_tool_context_parameter_with_default() -> None:
    def example_tool(
        context: ToolContext = DEFAULT_TOOL_CONTEXT,
    ) -> str:
        """Return the current customer."""
        return context.customer_id

    with pytest.raises(TypeError):
        tool(example_tool)


def test_tool_does_not_pass_tool_context_without_tool_context_parameter() -> None:
    @tool
    def example_tool(value: str) -> str:
        """Return the provided example value."""
        return value

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
        value: str,
    ) -> str:
        """Return the provided example value."""
        return value

    with pytest.raises(TypeError):
        example_tool({"value": 123})
