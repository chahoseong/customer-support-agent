from customer_support_agent.agent import Agent
from customer_support_agent.models import (
    ChatModel,
    ModelResponse,
    ToolCall,
)


class TestModel(ChatModel):
    __test__ = False

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._response = iter(responses)

    def generate(self, messages: object) -> ModelResponse:
        return next(self._response)


def test_agent_returns_final_text_without_tool_calls() -> None:
    model: ChatModel = TestModel(
        [
            ModelResponse(content="Your order is being processed."),
        ]
    )

    agent = Agent(model)

    result = agent.run("Where is my order?")

    assert result == "Your order is being processed."


def test_agent_continues_after_tool_call_and_returns_final_text() -> None:
    model: ChatModel = TestModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="get_order",
                        arguments={"order_id": "order-002"},
                    ),
                )
            ),
            ModelResponse(content="Your order has shipped."),
        ]
    )

    agent = Agent(model)

    result = agent.run("Where is order-002?")

    assert result == "Your order has shipped."
