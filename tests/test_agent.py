from collections.abc import Sequence

import pytest

from customer_support_agent.agent import Agent, AgentError
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools.tool import ToolContext, ToolDefinition, tool
from customer_support_agent.tools.toolset import Toolset


class ScriptedModel(ChatModel):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[ModelMessage, ...]] = []
        self.received_tools: list[tuple[ToolDefinition, ...]] = []

    def generate(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        self.requests.append(tuple(messages))
        self.received_tools.append(tuple(tools))
        return next(self._responses)


EMPTY_TOOLSET = Toolset(tools=())
TEST_CONTEXT = ToolContext(customer_id="customer-001")


def test_agent_returns_final_text_without_tool_calls() -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="Your order is being processed."),
        ]
    )

    agent = Agent(model, EMPTY_TOOLSET)

    result = agent.run("Where is my order?", context=TEST_CONTEXT)

    assert result == "Your order is being processed."


def test_agent_starts_each_run_with_new_message_history() -> None:
    first_request = ModelRequest(parts=(UserPromptPart(content="First request"),))
    second_request = ModelRequest(parts=(UserPromptPart(content="Second request"),))
    model = ScriptedModel(
        [
            ModelResponse(content="First response"),
            ModelResponse(content="Second response"),
        ]
    )
    agent = Agent(model, EMPTY_TOOLSET)

    first_result = agent.run("First request", context=TEST_CONTEXT)
    second_result = agent.run("Second request", context=TEST_CONTEXT)

    assert first_result == "First response"
    assert second_result == "Second response"
    assert model.requests == [
        (first_request,),
        (second_request,),
    ]


def test_agent_sends_tool_result_to_model_and_returns_final_text() -> None:
    @tool
    def configured_tool(
        context: ToolContext,
        value: str,
    ) -> dict[str, str]:
        """Return the configured value with its customer."""
        return {
            "customer_id": context.customer_id,
            "value": value,
        }

    toolset = Toolset(tools=(configured_tool,))
    context = ToolContext(customer_id="customer-001")

    user_request = ModelRequest(
        parts=(UserPromptPart(content="Use the configured tool."),)
    )
    tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name=configured_tool.definition.name,
                arguments={"value": "expected"},
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "customer_id": "customer-001",
                    "value": "expected",
                },
            ),
        )
    )
    final_response = ModelResponse(content="The configured value is expected.")
    model = ScriptedModel(
        [
            tool_call_response,
            final_response,
        ]
    )

    agent = Agent(model, toolset)

    result = agent.run("Use the configured tool.", context=context)

    assert result == "The configured value is expected."
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_preserves_message_history_across_two_tool_call_rounds() -> None:
    @tool
    def configured_tool(
        context: ToolContext,
        value: str,
    ) -> dict[str, str]:
        """Return the configured value with its customer."""
        return {
            "customer_id": context.customer_id,
            "value": value,
        }

    toolset = Toolset(tools=(configured_tool,))
    context = ToolContext(customer_id="customer-001")

    user_request = ModelRequest(
        parts=(
            UserPromptPart(
                content="Process the first and second values.",
            ),
        )
    )
    first_tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name=configured_tool.definition.name,
                arguments={"value": "first"},
            ),
        )
    )
    first_tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "customer_id": "customer-001",
                    "value": "first",
                },
            ),
        )
    )
    second_tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-2",
                name=configured_tool.definition.name,
                arguments={"value": "second"},
            ),
        )
    )
    second_tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-2",
                result={
                    "customer_id": "customer-001",
                    "value": "second",
                },
            ),
        )
    )
    final_response = ModelResponse(
        content="The first and second values were processed."
    )

    model = ScriptedModel(
        [
            first_tool_call_response,
            second_tool_call_response,
            final_response,
        ]
    )

    result = Agent(model, toolset).run(
        "Process the first and second values.", context=context
    )

    assert result == "The first and second values were processed."
    assert model.requests == [
        (user_request,),
        (
            user_request,
            first_tool_call_response,
            first_tool_result_request,
        ),
        (
            user_request,
            first_tool_call_response,
            first_tool_result_request,
            second_tool_call_response,
            second_tool_result_request,
        ),
    ]


def test_agent_prioritizes_tool_calls_when_response_also_has_content() -> None:
    @tool
    def configured_tool(
        context: ToolContext,
        value: str,
    ) -> dict[str, str]:
        """Return the configured value with its customer."""
        return {
            "customer_id": context.customer_id,
            "value": value,
        }

    toolset = Toolset(tools=(configured_tool,))

    user_message = "Process the configured value."
    user_request = ModelRequest(parts=(UserPromptPart(content=user_message),))

    tool_call_response = ModelResponse(
        content="The configured value is available.",
        tool_calls=(
            ToolCall(
                id="call-1",
                name=configured_tool.definition.name,
                arguments={"value": "expected"},
            ),
        ),
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "customer_id": "customer-001",
                    "value": "expected",
                },
            ),
        )
    )
    model = ScriptedModel(
        [
            tool_call_response,
            ModelResponse(content="The configured value is expected."),
        ]
    )

    result = Agent(model, toolset).run(user_message, context=TEST_CONTEXT)

    assert result == "The configured value is expected."
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        " \n\t ",
    ],
)
def test_agent_raises_invalid_model_response_for_content_without_displayable_text(
    content: str | None,
) -> None:
    model = ScriptedModel(
        [
            ModelResponse(content=content),
        ]
    )

    with pytest.raises(AgentError) as exc_info:
        Agent(model, EMPTY_TOOLSET).run(
            "Where is my order?",
            context=TEST_CONTEXT,
        )

    assert exc_info.value.code == "invalid_model_response"


def test_agent_returns_unknown_tool_error_to_model_and_continues() -> None:
    user_message = "Use an unavailable tool."
    user_request = ModelRequest(parts=(UserPromptPart(content=user_message),))

    tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name="unknown_tool",
                arguments={},
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "error": {
                        "code": "unknown_tool",
                        "message": (
                            "The requested tool is not available. "
                            "Use an available tool instead."
                        ),
                    }
                },
            ),
        )
    )

    model = ScriptedModel(
        [
            tool_call_response,
            ModelResponse(content="I cannot use that tool."),
        ]
    )

    result = Agent(model, EMPTY_TOOLSET).run(user_message, context=TEST_CONTEXT)

    assert result == "I cannot use that tool."
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_returns_tool_execution_failed_error_to_model_and_continues() -> None:
    @tool
    def broken_tool(
        value: str,
    ) -> object:
        """Raise runtime error"""
        raise RuntimeError("sensitive internal failure")

    toolset = Toolset(tools=(broken_tool,))

    user_message = "Complete the configured action."
    user_request = ModelRequest(parts=(UserPromptPart(content=user_message),))

    tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name=broken_tool.definition.name,
                arguments={"value": "expected"},
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "error": {
                        "code": "tool_execution_failed",
                        "message": (
                            "The tool failed unexpectedly; do not assume a result."
                        ),
                    }
                },
            ),
        )
    )

    model = ScriptedModel(
        [
            tool_call_response,
            ModelResponse(content="I could not complete the action."),
        ]
    )

    result = Agent(model, toolset).run(user_message, context=TEST_CONTEXT)

    assert result == "I could not complete the action."
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_wraps_model_failure_without_exposing_details() -> None:
    model_error = RuntimeError("sensitive provider failure")

    class FailingModel(ChatModel):
        def generate(
            self,
            _messages: Sequence[ModelMessage],
            _tools: Sequence[ToolDefinition],
        ) -> ModelResponse:
            raise model_error

    with pytest.raises(AgentError) as exc_info:
        Agent(FailingModel(), EMPTY_TOOLSET).run(
            "Where is my order?",
            context=TEST_CONTEXT,
        )

    assert exc_info.value.code == "model_call_failed"
    assert "sensitive provider failure" not in str(exc_info.value)
    assert exc_info.value.__cause__ is model_error


def test_agent_stops_before_executing_tool_from_fifth_model_call() -> None:
    tool_values = [f"value-{index}" for index in range(1, 6)]
    tool_arguments = [{"value": value} for value in tool_values]
    executed_values: list[str] = []

    @tool
    def recording_tool(value: str) -> dict[str, str]:
        """Record and return the configured value."""
        executed_values.append(value)
        return {"value": value}

    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{index}",
                        name=recording_tool.definition.name,
                        arguments=arguments,
                    ),
                )
            )
            for index, arguments in enumerate(tool_arguments, start=1)
        ]
    )

    toolset = Toolset(tools=(recording_tool,))

    with pytest.raises(AgentError) as exc_info:
        Agent(model, toolset).run("Process several values.", context=TEST_CONTEXT)

    assert exc_info.value.code == "model_call_limit_exceeded"
    assert len(model.requests) == 5
    assert executed_values == tool_values[:4]


def test_agent_accepts_final_text_from_fifth_model_call() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))
    tool_responses = [
        ModelResponse(
            tool_calls=(
                ToolCall(
                    id=f"call-{index}",
                    name=configured_tool.definition.name,
                    arguments={"value": f"value-{index}"},
                ),
            )
        )
        for index in range(1, 5)
    ]
    model = ScriptedModel(
        [
            *tool_responses,
            ModelResponse(content="The configured values were processed."),
        ]
    )

    result = Agent(model, toolset).run(
        "Process the configured values.",
        context=TEST_CONTEXT,
    )

    assert result == "The configured values were processed."
    assert len(model.requests) == 5


def test_agent_provides_configured_tool_definitions_on_every_model_call() -> None:
    @tool
    def configured_tool(
        context: ToolContext,
        value: str,
    ) -> str:
        """Return the provided example value."""
        return value

    toolset = Toolset(tools=(configured_tool,))
    context = ToolContext(customer_id="customer-001")

    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name=configured_tool.definition.name,
                        arguments={"value": "expected"},
                    ),
                )
            ),
            ModelResponse(content="The order is processing."),
        ]
    )

    Agent(model, toolset).run(
        "Use the configured tool.",
        context=context,
    )

    assert model.received_tools == [
        toolset.definitions,
        toolset.definitions,
    ]
