from collections.abc import Sequence

from pydantic import BaseModel

from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolResultPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools import ToolDefinition
from customer_support_agent.tools.errors import get_tool_error_code
from evals.order.models import (
    ObservedToolOutcome,
    ObservedToolUse,
)


def _interpret_tool_outcome(result: object) -> ObservedToolOutcome:
    tool_error_code = get_tool_error_code(result)

    if tool_error_code is not None:
        return tool_error_code

    if type(result) is dict and "error" in result:
        return "uninterpretable"

    return "success"


class ToolUseRecordingModel(ChatModel):
    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self._pending_tool_calls: dict[str, ToolCallPart] = {}
        self._tool_uses: list[ObservedToolUse] = []

    @property
    def tool_uses(self) -> tuple[ObservedToolUse, ...]:
        return tuple(self._tool_uses)

    def _record_completed_tool_uses(
        self,
        messages: Sequence[ModelMessage],
    ) -> None:
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue

            for part in message.parts:
                if not isinstance(part, ToolResultPart):
                    continue

                tool_call = self._pending_tool_calls.pop(part.tool_call_id, None)
                if tool_call is None:
                    continue

                self._tool_uses.append(
                    ObservedToolUse(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        outcome=_interpret_tool_outcome(part.result),
                    )
                )

    def _store_pending_tool_calls(self, response: ModelResponse) -> None:
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                self._pending_tool_calls[part.id] = part

    def generate(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        *,
        output_type: type[BaseModel] | None = None,
        instructions: str | None = None,
    ) -> ModelResponse:
        self._record_completed_tool_uses(messages)

        response = self._model.generate(
            messages,
            tools,
            output_type=output_type,
            instructions=instructions,
        )

        self._store_pending_tool_calls(response)

        return response
