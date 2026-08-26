from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from customer_support_agent.agent import Agent, AgentError, AgentResult
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StructuredOutputPart,
    ToolCallPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools.tool import ToolContext, ToolDefinition, tool
from customer_support_agent.tools.toolset import Toolset
from tests.scripted_model import ScriptedModel

EMPTY_TOOLSET = Toolset(tools=())
TEST_CONTEXT = ToolContext(customer_id="customer-001")


def test_agent_returns_agent_result_without_tool_calls() -> None:
    expected_result = AgentResult(
        message="Your order is being processed.",
    )

    model = ScriptedModel(
        [
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    result = Agent(model, EMPTY_TOOLSET).run(
        "Where is my order?",
        context=TEST_CONTEXT,
    )

    assert result == expected_result


def test_agent_starts_each_run_with_new_message_history() -> None:
    first_request = ModelRequest(parts=(UserPromptPart(content="First request"),))
    second_request = ModelRequest(parts=(UserPromptPart(content="Second request"),))
    first_expected_result = AgentResult(message="First response")
    second_expected_result = AgentResult(message="Second response")

    model = ScriptedModel(
        [
            ModelResponse(parts=(StructuredOutputPart(output=first_expected_result),)),
            ModelResponse(parts=(StructuredOutputPart(output=second_expected_result),)),
        ]
    )
    agent = Agent(model, EMPTY_TOOLSET)

    first_result = agent.run("First request", context=TEST_CONTEXT)
    second_result = agent.run("Second request", context=TEST_CONTEXT)

    assert first_result == first_expected_result
    assert second_result == second_expected_result
    assert model.requests == [
        (first_request,),
        (second_request,),
    ]


class UnexpectedOutput(BaseModel):
    value: str


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            ModelResponse(),
            id="without-parts",
        ),
        pytest.param(
            ModelResponse(
                parts=(
                    StructuredOutputPart(
                        output=AgentResult(message="First result"),
                    ),
                    StructuredOutputPart(
                        output=AgentResult(message="Second result"),
                    ),
                )
            ),
            id="multiple-agent-results",
        ),
        pytest.param(
            ModelResponse(
                parts=(
                    StructuredOutputPart(
                        output=UnexpectedOutput(value="unexpected"),
                    ),
                )
            ),
            id="unexpected-output-type",
        ),
    ],
)
def test_agent_raises_invalid_model_response_without_exactly_one_agent_result(
    response: ModelResponse,
) -> None:
    model = ScriptedModel([response])

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
            *,
            output_type: type[BaseModel],
            instructions: str | None = None,
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
                parts=(
                    ToolCallPart(
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


def test_agent_accepts_agent_result_from_fifth_model_call() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))

    tool_responses = [
        ModelResponse(
            parts=(
                ToolCallPart(
                    id=f"call-{index}",
                    name=configured_tool.definition.name,
                    arguments={"value": f"value-{index}"},
                ),
            )
        )
        for index in range(1, 5)
    ]

    expected_result = AgentResult(
        message="The configured values were processed.",
    )

    model = ScriptedModel(
        [
            *tool_responses,
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    result = Agent(model, toolset).run(
        "Process the configured values.",
        context=TEST_CONTEXT,
    )

    assert result == expected_result
    assert len(model.requests) == 5


def test_agent_provides_configured_instructions_on_every_model_call() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=configured_tool.definition.name,
                        arguments={"value": "expected"},
                    ),
                )
            ),
            ModelResponse(
                parts=(
                    StructuredOutputPart(
                        output=AgentResult(
                            message="The configured value was processed.",
                        )
                    ),
                )
            ),
        ]
    )

    instructions = "Follow the configured instructions."

    Agent(
        model,
        toolset,
        instructions=instructions,
    ).run(
        "Process the configured value.",
        context=TEST_CONTEXT,
    )

    assert model.received_instructions == [
        instructions,
        instructions,
    ]


def test_agent_provides_agent_result_type_on_every_model_call() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=configured_tool.definition.name,
                        arguments={"value": "expected"},
                    ),
                )
            ),
            ModelResponse(
                parts=(
                    StructuredOutputPart(
                        output=AgentResult(
                            message="The configured value was processed.",
                        )
                    ),
                )
            ),
        ]
    )

    Agent(
        model,
        toolset,
    ).run(
        "Process the configured value.",
        context=TEST_CONTEXT,
    )

    assert model.received_output_types == [
        AgentResult,
        AgentResult,
    ]
