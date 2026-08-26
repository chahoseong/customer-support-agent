from abc import ABC, abstractmethod
from collections.abc import Sequence

from customer_support_agent.messages import ModelMessage, ModelResponse
from customer_support_agent.tools import ToolDefinition


class ChatModel(ABC):
    @abstractmethod
    def generate(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        *,
        instructions: str | None = None,
    ) -> ModelResponse:
        raise NotImplementedError
