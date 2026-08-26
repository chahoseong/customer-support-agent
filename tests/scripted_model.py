from collections.abc import Sequence

from pydantic import BaseModel

from customer_support_agent.messages import ModelMessage, ModelResponse
from customer_support_agent.models import ChatModel
from customer_support_agent.tools.tool import ToolDefinition


class ScriptedModel(ChatModel):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[ModelMessage, ...]] = []
        self.received_tools: list[tuple[ToolDefinition, ...]] = []
        self.received_instructions: list[str | None] = []
        self.received_output_types: list[type[BaseModel]] = []

    def generate(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        *,
        output_type: type[BaseModel],
        instructions: str | None = None,
    ) -> ModelResponse:
        self.requests.append(tuple(messages))
        self.received_tools.append(tuple(tools))
        self.received_instructions.append(instructions)
        self.received_output_types.append(output_type)
        return next(self._responses)
