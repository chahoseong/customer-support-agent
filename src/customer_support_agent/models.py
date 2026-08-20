from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelResponse:
    content: str


class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages: object) -> ModelResponse:
        raise NotImplementedError
