from customer_support_agent.get_order import get_order
from customer_support_agent.models import ChatModel


class Agent:
    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def run(self, user_message: str) -> str:
        messages = [
            {
                "role": "user",
                "content": user_message,
            }
        ]

        while True:
            response = self._model.generate(messages)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.name != "get_order":
                        raise ValueError(f"Unsupported tool: {tool_call.name}")

                    get_order(tool_call.arguments)

                continue

            if response.content is None:
                raise ValueError("Model response has no content.")

            return response.content
