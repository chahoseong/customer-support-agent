from collections.abc import Sequence

import pytest

from customer_support_agent.agent import Agent, AgentError
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools.tool import ToolContext, ToolDefinition, tool
from customer_support_agent.tools.toolset import Toolset
from tests.scripted_model import ScriptedModel

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
