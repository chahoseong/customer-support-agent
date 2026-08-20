from collections.abc import Sequence

from customer_support_agent.agent import Agent
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel


class TestModel(ChatModel):
    __test__ = False

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._response = iter(responses)
        self.requests: list[tuple[ModelMessage, ...]] = []

    def generate(self, messages: Sequence[ModelMessage]) -> ModelResponse:
        self.requests.append(tuple(messages))
        return next(self._response)


def test_agent_returns_final_text_without_tool_calls() -> None:
    model = TestModel(
        [
            ModelResponse(content="Your order is being processed."),
        ]
    )

    agent = Agent(model)

    result = agent.run("Where is my order?")

    assert result == "Your order is being processed."


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
    model = TestModel(
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
