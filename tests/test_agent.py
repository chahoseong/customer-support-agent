from customer_support_agent.agent import Agent
from customer_support_agent.models import ChatModel, ModelResponse


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
