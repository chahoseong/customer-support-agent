from collections.abc import Sequence

import pytest

import customer_support_agent.agent as agent_module
from customer_support_agent.agent import Agent, AgentRunError
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel


class ScriptedModel(ChatModel):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[ModelMessage, ...]] = []

    def generate(self, messages: Sequence[ModelMessage]) -> ModelResponse:
        self.requests.append(tuple(messages))
        return next(self._responses)


def test_agent_returns_final_text_without_tool_calls() -> None:
    model = ScriptedModel(
        [
            ModelResponse(content="Your order is being processed."),
        ]
    )

    agent = Agent(model)

    result = agent.run("Where is my order?")

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
    agent = Agent(model)

    first_result = agent.run("First request")
    second_result = agent.run("Second request")

    assert first_result == "First response"
    assert second_result == "Second response"
    assert model.requests == [
        (first_request,),
        (second_request,),
    ]


def test_agent_sends_tool_result_to_model_and_returns_final_text() -> None:
    user_request = ModelRequest(parts=(UserPromptPart(content="Where is order-002?"),))
    tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name="get_order",
                arguments={"order_id": "order-002"},
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "order_id": "order-002",
                    "status": "shipped",
                },
            ),
        )
    )
    final_response = ModelResponse(content="Your order has shipped.")
    model = ScriptedModel(
        [
            tool_call_response,
            final_response,
        ]
    )

    agent = Agent(model)

    result = agent.run("Where is order-002?")

    assert result == "Your order has shipped."
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_handles_two_tool_call_rounds_before_final_response() -> None:
    user_request = ModelRequest(
        parts=(
            UserPromptPart(
                content="Compare order-001 and order-002.",
            ),
        )
    )
    first_tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name="get_order",
                arguments={"order_id": "order-001"},
            ),
        )
    )
    first_tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "order_id": "order-001",
                    "status": "processing",
                },
            ),
        )
    )
    second_tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-2",
                name="get_order",
                arguments={"order_id": "order-002"},
            ),
        )
    )
    second_tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-2",
                result={
                    "order_id": "order-002",
                    "status": "shipped",
                },
            ),
        )
    )
    final_response = ModelResponse(
        content="Order-001 is processing and order-002 has shipped."
    )

    model = ScriptedModel(
        [
            first_tool_call_response,
            second_tool_call_response,
            final_response,
        ]
    )

    result = Agent(model).run("Compare order-001 and order-002.")

    assert result == "Order-001 is processing and order-002 has shipped."
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
    tool_call_response = ModelResponse(
        content="Let me check that order.",
        tool_calls=(
            ToolCall(
                id="call-1",
                name="get_order",
                arguments={"order_id": "order-002"},
            ),
        ),
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "order_id": "order-002",
                    "status": "shipped",
                },
            ),
        )
    )
    model = ScriptedModel(
        [
            tool_call_response,
            ModelResponse(content="Your order has shipped."),
        ]
    )

    result = Agent(model).run("Where is order-002?")

    assert result == "Your order has shipped."
    assert model.requests[1][-2:] == (
        tool_call_response,
        tool_result_request,
    )


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

    with pytest.raises(AgentRunError) as exc_info:
        Agent(model).run("Where is my order?")

    assert exc_info.value.code == "invalid_model_response"


def test_agent_returns_unknown_tool_error_to_model_and_continues() -> None:
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

    result = Agent(model).run("Send a message.")

    assert result == "I cannot use that tool."

    second_model_messages = model.requests[1]

    assert second_model_messages[-2:] == (
        tool_call_response,
        tool_result_request,
    )


def test_agent_returns_unexpected_tool_failure_to_model_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_order(_arguments: object) -> object:
        raise RuntimeError("sensitive internal failure")

    monkeypatch.setattr(agent_module, "get_order", fail_get_order)

    tool_call_response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name="get_order",
                arguments={"order_id": "order-001"},
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
            ModelResponse(content="I could not retrieve the order."),
        ]
    )

    result = Agent(model).run("Where is order-001?")

    assert result == "I could not retrieve the order."

    second_model_messages = model.requests[1]

    assert second_model_messages[-2:] == (
        tool_call_response,
        tool_result_request,
    )


def test_agent_wraps_model_failure_without_exposing_details() -> None:
    model_error = RuntimeError("sensitive provider failure")

    class FailingModel(ChatModel):
        def generate(
            self,
            _messages: Sequence[ModelMessage],
        ) -> ModelResponse:
            raise model_error

    with pytest.raises(AgentRunError) as exc_info:
        Agent(FailingModel()).run("Where is my order?")

    assert exc_info.value.code == "model_call_failed"
    assert "sensitive provider failure" not in str(exc_info.value)
    assert exc_info.value.__cause__ is model_error


def test_agent_stops_before_executing_tool_from_fifth_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_arguments = [{"order_id": f"order-{index:03d}"} for index in range(1, 6)]
    executed_arguments: list[object] = []

    def record_get_order(arguments: object) -> object:
        executed_arguments.append(arguments)
        return {"status": "recorded"}

    monkeypatch.setattr(agent_module, "get_order", record_get_order)

    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{index}",
                        name="get_order",
                        arguments=arguments,
                    ),
                )
            )
            for index, arguments in enumerate(tool_arguments, start=1)
        ]
    )

    with pytest.raises(AgentRunError) as exc_info:
        Agent(model).run("Check several orders.")

    assert exc_info.value.code == "model_call_limit_exceeded"
    assert len(model.requests) == 5
    assert executed_arguments == tool_arguments[:4]


def test_agent_accepts_final_text_from_fifth_model_call() -> None:
    tool_responses = [
        ModelResponse(
            tool_calls=(
                ToolCall(
                    id=f"call-{index}",
                    name="get_order",
                    arguments={"order_id": "order-001"},
                ),
            )
        )
        for index in range(1, 5)
    ]
    model = ScriptedModel(
        [
            *tool_responses,
            ModelResponse(content="The order is still processing."),
        ]
    )

    result = Agent(model).run("Keep checking the order.")

    assert result == "The order is still processing."
    assert len(model.requests) == 5
