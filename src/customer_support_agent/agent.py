from typing import Literal

from customer_support_agent.get_order import get_order
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tool_errors import create_tool_error

_MAX_MODEL_CALLS = 5

type AgentRunErrorCode = Literal[
    "invalid_model_response",
    "model_call_failed",
    "model_call_limit_exceeded",
]


_AGENT_RUN_ERROR_MESSAGES: dict[AgentRunErrorCode, str] = {
    "invalid_model_response": "Model response has no displayable content.",
    "model_call_failed": "Model call failed.",
    "model_call_limit_exceeded": "Model call limit exceeded.",
}


class AgentRunError(RuntimeError):
    def __init__(self, code: AgentRunErrorCode) -> None:
        self.code = code
        super().__init__(_AGENT_RUN_ERROR_MESSAGES[code])


class Agent:
    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def run(self, user_message: str) -> str:
        messages: list[ModelMessage] = [
            ModelRequest(parts=(UserPromptPart(content=user_message),))
        ]
        model_call_count = 0

        while True:
            model_call_count += 1

            try:
                response = self._model.generate(messages)
            except Exception as error:
                raise AgentRunError("model_call_failed") from error

            if response.tool_calls:
                if model_call_count >= _MAX_MODEL_CALLS:
                    raise AgentRunError("model_call_limit_exceeded")

                messages.append(response)
                tool_results: list[ToolResultPart] = []

                for tool_call in response.tool_calls:
                    result: object

                    if tool_call.name != "get_order":
                        result = create_tool_error("unknown_tool")
                    else:
                        try:
                            result = get_order(tool_call.arguments)
                        except Exception:
                            result = create_tool_error("tool_execution_failed")

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

            if response.content is None or not response.content.strip():
                raise AgentRunError("invalid_model_response")

            return response.content
