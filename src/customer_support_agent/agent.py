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
        response = self._model.generate(messages)
        return response.content
