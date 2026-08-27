from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from customer_support_agent.agent import Agent, AgentError, AgentResult
from customer_support_agent.conversation import (
    AgentMessage,
    Conversation,
    UserMessage,
)
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools.tool import ToolContext, ToolDefinition, tool
from customer_support_agent.tools.toolset import Toolset
from tests.scripted_model import ScriptedModel

EMPTY_TOOLSET = Toolset(tools=())
TEST_CONTEXT = ToolContext(customer_id="customer-001")


def test_agent_returns_structured_result_after_model_stops_calling_tools() -> None:
    @tool
    def configured_tool(value: str) -> str:
        """Return the configured value."""
        return value

    toolset = Toolset(tools=(configured_tool,))
    user_request = ModelRequest(parts=(UserPromptPart(content="Where is my order?"),))
    expected_result = AgentResult(
        message="Your order is being processed.",
    )

    model = ScriptedModel(
        [
            ModelResponse(parts=(TextPart(content="Your order is being processed."),)),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    result = Agent(model, toolset).run(
        "Where is my order?",
        context=TEST_CONTEXT,
    )

    assert result == expected_result
    assert model.requests == [
        (user_request,),
        (user_request,),
    ]
    assert model.received_tools == [
        toolset.definitions,
        (),
    ]
    assert model.received_output_types == [
        None,
        AgentResult,
    ]


def test_agent_uses_only_conversation_passed_to_each_run() -> None:
    first_conversation = Conversation(
        messages=(
            UserMessage(content="Where is my shipped order?"),
            AgentMessage(content="Please provide the shipped order ID."),
        )
    )

    second_conversation = Conversation(
        messages=(
            UserMessage(content="Can I cancel my order?"),
            AgentMessage(content="Please provide the order ID you want to cancel."),
        )
    )

    first_user_message = "The order ID is order-002."
    second_user_message = "The order ID is order-004."

    first_expected_result = AgentResult(message="First response")
    second_expected_result = AgentResult(message="Second response")

    model = ScriptedModel(
        [
            ModelResponse(parts=(TextPart(content=first_expected_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=first_expected_result),)),
            ModelResponse(parts=(TextPart(content=second_expected_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=second_expected_result),)),
        ]
    )
    agent = Agent(model, EMPTY_TOOLSET)

    agent.run(
        first_user_message,
        context=TEST_CONTEXT,
        conversation=first_conversation,
    )
    agent.run(
        second_user_message,
        context=TEST_CONTEXT,
        conversation=second_conversation,
    )

    assert model.requests == [
        (
            ModelRequest(parts=(UserPromptPart(content="Where is my shipped order?"),)),
            ModelResponse(
                parts=(TextPart(content="Please provide the shipped order ID."),)
            ),
            ModelRequest(parts=(UserPromptPart(content=first_user_message),)),
        ),
        (
            ModelRequest(parts=(UserPromptPart(content="Where is my shipped order?"),)),
            ModelResponse(
                parts=(TextPart(content="Please provide the shipped order ID."),)
            ),
            ModelRequest(parts=(UserPromptPart(content=first_user_message),)),
        ),
        (
            ModelRequest(parts=(UserPromptPart(content="Can I cancel my order?"),)),
            ModelResponse(
                parts=(
                    TextPart(content="Please provide the order ID you want to cancel."),
                )
            ),
            ModelRequest(parts=(UserPromptPart(content=second_user_message),)),
        ),
        (
            ModelRequest(parts=(UserPromptPart(content="Can I cancel my order?"),)),
            ModelResponse(
                parts=(
                    TextPart(content="Please provide the order ID you want to cancel."),
                )
            ),
            ModelRequest(parts=(UserPromptPart(content=second_user_message),)),
        ),
    ]


def test_agent_excludes_tool_history_from_follow_up_run() -> None:
    @tool
    def get_available_orders() -> list[str]:
        """Return the available order identifiers."""
        return ["order-001", "order-002"]

    clarification_result = AgentResult(
        message="Which order do you mean?",
    )
    final_result = AgentResult(
        message="Order order-002 has shipped.",
    )

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=get_available_orders.definition.name,
                        arguments={},
                    ),
                )
            ),
            ModelResponse(parts=(TextPart(content=clarification_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=clarification_result),)),
            ModelResponse(parts=(TextPart(content=final_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=final_result),)),
        ]
    )
    agent = Agent(model, Toolset(tools=(get_available_orders,)))

    first_user_message = "Which of my orders has shipped?"
    first_result = agent.run(first_user_message, context=TEST_CONTEXT)

    conversation = Conversation(
        messages=(
            UserMessage(content=first_user_message),
            AgentMessage(content=first_result.message),
        )
    )

    agent.run(
        "order-002",
        context=TEST_CONTEXT,
        conversation=conversation,
    )

    assert model.requests[3] == (
        ModelRequest(parts=(UserPromptPart(content=first_user_message),)),
        ModelResponse(parts=(TextPart(content="Which order do you mean?"),)),
        ModelRequest(parts=(UserPromptPart(content="order-002"),)),
    )


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
    model = ScriptedModel(
        [
            ModelResponse(parts=(TextPart(content="Draft response"),)),
            response,
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
            *,
            output_type: type[BaseModel] | None = None,
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


def test_agent_stops_before_final_response_call_when_fifth_model_call_returns_text() -> (
    None
):
    tool_values = [f"value-{index}" for index in range(1, 5)]
    executed_values: list[str] = []

    @tool
    def recording_tool(value: str) -> str:
        """Record and return the configured value."""
        executed_values.append(value)
        return value

    toolset = Toolset(tools=(recording_tool,))
    tool_responses = [
        ModelResponse(
            parts=(
                ToolCallPart(
                    id=f"call-{index}",
                    name=recording_tool.definition.name,
                    arguments={"value": value},
                ),
            )
        )
        for index, value in enumerate(tool_values, start=1)
    ]
    model = ScriptedModel(
        [
            *tool_responses,
            ModelResponse(parts=(TextPart(content="The values were processed."),)),
        ]
    )

    with pytest.raises(AgentError) as exc_info:
        Agent(model, toolset).run(
            "Process several values.",
            context=TEST_CONTEXT,
        )

    assert exc_info.value.code == "model_call_limit_exceeded"
    assert len(model.requests) == 5
    assert model.received_output_types == [None] * 5
    assert executed_values == tool_values


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
        for index in range(1, 4)
    ]

    expected_result = AgentResult(
        message="The configured values were processed.",
    )

    model = ScriptedModel(
        [
            *tool_responses,
            ModelResponse(parts=(TextPart(content=expected_result.message),)),
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
                    TextPart(
                        content="The configured value was processed.",
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
        instructions,
    ]


def test_agent_requests_agent_result_type_only_for_final_model_call() -> None:
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
                    TextPart(
                        content="The configured value was processed.",
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
        None,
        None,
        AgentResult,
    ]
