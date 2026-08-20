from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: object


@dataclass
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages: object) -> ModelResponse:
        raise NotImplementedError
