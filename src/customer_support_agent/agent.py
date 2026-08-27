import logging
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from customer_support_agent.conversation import (
    Conversation,
    UserMessage,
)
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools import ToolContext, Toolset

logger = logging.getLogger(__name__)

_MAX_MODEL_CALLS = 5
_EMPTY_CONVERSATION = Conversation()

type AgentErrorCode = Literal[
    "invalid_model_response",
    "model_call_failed",
    "model_call_limit_exceeded",
]


_AGENT_ERROR_MESSAGES: dict[AgentErrorCode, str] = {
    "invalid_model_response": "Model response has no valid AgentResult.",
    "model_call_failed": "Model call failed.",
    "model_call_limit_exceeded": "Model call limit exceeded.",
}


class AgentResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    message: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
        ),
    ]


class AgentError(RuntimeError):
    def __init__(self, code: AgentErrorCode) -> None:
        self.code = code
        super().__init__(_AGENT_ERROR_MESSAGES[code])


class Agent:
    def __init__(
        self, model: ChatModel, toolset: Toolset, *, instructions: str | None = None
    ) -> None:
        self._model = model
        self._toolset = toolset
        self._instructions = instructions

    def run(
        self,
        user_message: str,
        *,
        context: ToolContext,
        conversation: Conversation = _EMPTY_CONVERSATION,
    ) -> AgentResult:
        messages: list[ModelMessage] = []

        for message in conversation.messages:
            if isinstance(message, UserMessage):
                messages.append(
                    ModelRequest(parts=(UserPromptPart(content=message.content),))
                )
            else:
                messages.append(
                    ModelResponse(parts=(TextPart(content=message.content),))
                )

        messages.append(ModelRequest(parts=(UserPromptPart(content=user_message),)))

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
                    instructions=self._instructions,
                )
            except Exception as error:
                logger.error(
                    "The run stopped because the model call failed.",
                )

                raise AgentError("model_call_failed") from error

            tool_calls = tuple(
                part for part in response.parts if isinstance(part, ToolCallPart)
            )

            if tool_calls:
                if model_call_count >= _MAX_MODEL_CALLS:
                    logger.error(
                        "The run stopped because another model call would exceed "
                        "the limit of %d.",
                        _MAX_MODEL_CALLS,
                    )

                    raise AgentError("model_call_limit_exceeded")

                messages.append(response)
                tool_results: list[ToolResultPart] = []

                for tool_call in tool_calls:
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

            if len(response.parts) != 1 or not isinstance(response.parts[0], TextPart):
                logger.error(
                    "The run stopped because the model returned no valid text response."
                )

                raise AgentError("invalid_model_response")

            if model_call_count >= _MAX_MODEL_CALLS:
                logger.error(
                    "The run stopped because another model call would exceed "
                    "the limit of %d.",
                    _MAX_MODEL_CALLS,
                )

                raise AgentError("model_call_limit_exceeded")

            model_call_count += 1

            logger.info(
                "Calling the model (%d of %d) for the final response.",
                model_call_count,
                _MAX_MODEL_CALLS,
            )

            try:
                final_response = self._model.generate(
                    messages,
                    (),
                    instructions=self._instructions,
                    output_type=AgentResult,
                )
            except Exception as error:
                logger.error(
                    "The run stopped because the model call failed.",
                )

                raise AgentError("model_call_failed") from error

            if len(final_response.parts) == 1:
                part = final_response.parts[0]

                if isinstance(part, StructuredOutputPart) and isinstance(
                    part.output, AgentResult
                ):
                    logger.info("The run finished with the model's final response.")
                    return part.output

            logger.error(
                "The run stopped because the model returned no valid AgentResult."
            )

            raise AgentError("invalid_model_response")
