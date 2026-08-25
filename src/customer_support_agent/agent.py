import logging
from typing import Literal

from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools import ToolContext, Toolset

logger = logging.getLogger(__name__)

_MAX_MODEL_CALLS = 5

type AgentErrorCode = Literal[
    "invalid_model_response",
    "model_call_failed",
    "model_call_limit_exceeded",
]


_AGENT_ERROR_MESSAGES: dict[AgentErrorCode, str] = {
    "invalid_model_response": "Model response has no displayable content.",
    "model_call_failed": "Model call failed.",
    "model_call_limit_exceeded": "Model call limit exceeded.",
}


class AgentError(RuntimeError):
    def __init__(self, code: AgentErrorCode) -> None:
        self.code = code
        super().__init__(_AGENT_ERROR_MESSAGES[code])


class Agent:
    def __init__(self, model: ChatModel, toolset: Toolset) -> None:
        self._model = model
        self._toolset = toolset

    def run(self, user_message: str, *, context: ToolContext) -> str:
        messages: list[ModelMessage] = [
            ModelRequest(parts=(UserPromptPart(content=user_message),))
        ]
        model_call_count = 0

        while True:
            model_call_count += 1

            logger.info(
                "Calling the model (%d of %d).",
                model_call_count,
                _MAX_MODEL_CALLS,
            )

            try:
                response = self._model.generate(
                    messages,
                    self._toolset.definitions,
                )
            except Exception as error:
                logger.error(
                    "The run stopped because the model call failed.",
                )

                raise AgentError("model_call_failed") from error

            if response.tool_calls:
                if model_call_count >= _MAX_MODEL_CALLS:
                    logger.error(
                        "The run stopped because another model call would exceed "
                        "the limit of %d.",
                        _MAX_MODEL_CALLS,
                    )

                    raise AgentError("model_call_limit_exceeded")

                messages.append(response)
                tool_results: list[ToolResultPart] = []

                for tool_call in response.tool_calls:
                    logger.info(
                        "The model requested the %r tool.",
                        tool_call.name,
                    )
                    logger.info(" ⨽ Call ID: %r", tool_call.id)
                    logger.info(" ⨽ Arguments: %r", tool_call.arguments)

                    result = self._toolset.execute(
                        tool_call.name,
                        tool_call.arguments,
                        context=context,
                    )

                    logger.info(" ⨽ Result: %r", result)

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

                logger.info(
                    "Calling the model again with the tool results.",
                )

                continue

            if response.content is None or not response.content.strip():
                logger.error(
                    "The run stopped because the model returned no displayable content."
                )

                raise AgentError("invalid_model_response")

            logger.info("The run finished with the model's final response.")

            return response.content
