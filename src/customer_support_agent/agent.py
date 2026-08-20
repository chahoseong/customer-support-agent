from customer_support_agent.get_order import get_order
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel


class Agent:
    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def run(self, user_message: str) -> str:
        messages: list[ModelMessage] = [
            ModelRequest(parts=(UserPromptPart(content=user_message),))
        ]

        while True:
            response = self._model.generate(messages)

            if response.tool_calls:
                messages.append(response)
                tool_results: list[ToolResultPart] = []

                for tool_call in response.tool_calls:
                    if tool_call.name != "get_order":
                        raise ValueError(f"Unsupported tool: {tool_call.name}")

                    result = get_order(tool_call.arguments)
                    tool_results.append(
                        ToolResultPart(
                            tool_call_id=tool_call.id,
                            result=result,
                        )
                    )

                messages.append(
                    ModelRequest(
                        parts=tuple(tool_results),
                    )
                )
                continue

            if response.content is None:
                raise ValueError("Model response has no content.")

            return response.content
