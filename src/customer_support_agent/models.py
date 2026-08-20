from abc import ABC, abstractmethod
from collections.abc import Sequence

from customer_support_agent.messages import ModelMessage, ModelResponse


class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages: Sequence[ModelMessage]) -> ModelResponse:
        raise NotImplementedError
