import logging
from dataclasses import dataclass
from typing import Annotated, Literal

import logfire
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
from customer_support_agent.tools import (
    ToolContext,
    ToolDefinition,
    Toolset,
)

logger = logging.getLogger(__name__)

_MAX_MODEL_CALLS = 5
_EMPTY_CONVERSATION = Conversation()

type AgentErrorCode = Literal[
    "invalid_model_response",
    "model_call_failed",
    "model_call_limit_exceeded",
]

type ModelCallPurpose = Literal[
    "agent_loop",
    "final_result",
]

type ModelResponseKind = Literal[
    "tool_calls",
    "text",
    "agent_result",
    "invalid",
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


def _classify_model_response(response: ModelResponse) -> ModelResponseKind:
    if any(isinstance(part, ToolCallPart) for part in response.parts):
        return "tool_calls"

    if len(response.parts) == 1 and isinstance(response.parts[0], TextPart):
        return "text"

    if (
        len(response.parts) == 1
        and isinstance(response.parts[0], StructuredOutputPart)
        and isinstance(response.parts[0].output, AgentResult)
    ):
        return "agent_result"

    return "invalid"


class AgentError(RuntimeError):
    def __init__(self, code: AgentErrorCode) -> None:
        self.code = code
        super().__init__(_AGENT_ERROR_MESSAGES[code])


@dataclass(frozen=True)
class _AgentLoopResult:
    value: AgentResult | Exception
    model_call_count: int
    tool_call_count: int


class Agent:
    def __init__(
        self, model: ChatModel, toolset: Toolset, *, instructions: str | None = None
    ) -> None:
        self._model = model
        self._toolset = toolset
        self._instructions = instructions

    def _generate_model_response(
        self,
        messages: list[ModelMessage],
        tools: tuple[ToolDefinition, ...],
        *,
        call_index: int,
        purpose: ModelCallPurpose,
        output_type: type[BaseModel] | None = None,
    ) -> ModelResponse:
        model_error: Exception | None = None

        with logfire.span("model.generate") as model_span:
            model_span.set_attributes(
                {
                    "customer_support_agent.model.call.index": call_index,
                    "customer_support_agent.model.call.purpose": purpose,
                }
            )

            try:
                response = self._model.generate(
                    messages,
                    tools,
                    output_type=output_type,
                    instructions=self._instructions,
                )
            except Exception as error:
                model_error = error
                model_span.set_attributes(
                    {
                        "customer_support_agent.model.outcome": "exception",
                        "error.type": type(error).__name__,
                    }
                )
                model_span.set_level("error")
            else:
                tool_call_count = sum(
                    isinstance(part, ToolCallPart) for part in response.parts
                )
                response_kind = _classify_model_response(response)

                model_span.set_attributes(
                    {
                        "customer_support_agent.model.outcome": "success",
                        "customer_support_agent.model.response.kind": response_kind,
                        "customer_support_agent.model.tool_call.count": tool_call_count,
                    }
                )

                return response

        assert model_error is not None
        logger.error(
            "The run stopped because the model call failed.",
        )
        raise AgentError("model_call_failed") from model_error

    def _execute_tool(
        self,
        tool_call: ToolCallPart,
        *,
        context: ToolContext,
    ) -> ToolResultPart:
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

        return ToolResultPart(
            tool_call_id=tool_call.id,
            result=result,
        )

    def _generate_final_result(
        self,
        messages: list[ModelMessage],
        *,
        call_index: int,
    ) -> AgentResult:
        logger.info(
            "Calling the model (%d of %d) for the final response.",
            call_index,
            _MAX_MODEL_CALLS,
        )

        response = self._generate_model_response(
            messages,
            (),
            call_index=call_index,
            purpose="final_result",
            output_type=AgentResult,
        )

        if _classify_model_response(response) == "agent_result":
            part = response.parts[0]
            assert isinstance(part, StructuredOutputPart)
            assert isinstance(part.output, AgentResult)

            logger.info("The run finished with the model's final response.")
            return part.output

        logger.error("The run stopped because the model returned no valid AgentResult.")
        raise AgentError("invalid_model_response")

    def _build_model_messages(
        self,
        user_message: str,
        conversation: Conversation,
    ) -> list[ModelMessage]:
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
        return messages

    def _ensure_model_call_is_within_limit(self, call_index: int) -> None:
        if call_index <= _MAX_MODEL_CALLS:
            return

        logger.error(
            "The run stopped because another model call would exceed the limit of %d.",
            _MAX_MODEL_CALLS,
        )
        raise AgentError("model_call_limit_exceeded")

    def _execute_agent_loop(
        self,
        user_message: str,
        *,
        context: ToolContext,
        conversation: Conversation,
    ) -> _AgentLoopResult:
        model_call_count = 0
        tool_call_count = 0

        try:
            messages = self._build_model_messages(user_message, conversation)
            while True:
                model_call_count += 1

                logger.info(
                    "Calling the model (%d of %d).",
                    model_call_count,
                    _MAX_MODEL_CALLS,
                )

                response = self._generate_model_response(
                    messages,
                    self._toolset.definitions,
                    call_index=model_call_count,
                    purpose="agent_loop",
                )
                response_kind = _classify_model_response(response)

                if response_kind == "tool_calls":
                    tool_calls = tuple(
                        part
                        for part in response.parts
                        if isinstance(part, ToolCallPart)
                    )
                    self._ensure_model_call_is_within_limit(model_call_count + 1)

                    messages.append(response)
                    tool_results: list[ToolResultPart] = []

                    for tool_call in tool_calls:
                        tool_call_count += 1
                        tool_results.append(
                            self._execute_tool(
                                tool_call,
                                context=context,
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

                if response_kind != "text":
                    logger.error(
                        "The run stopped because the model returned no valid text response."
                    )

                    raise AgentError("invalid_model_response")

                self._ensure_model_call_is_within_limit(model_call_count + 1)

                model_call_count += 1
                result = self._generate_final_result(
                    messages,
                    call_index=model_call_count,
                )
                return _AgentLoopResult(
                    value=result,
                    model_call_count=model_call_count,
                    tool_call_count=tool_call_count,
                )
        except Exception as error:
            return _AgentLoopResult(
                value=error,
                model_call_count=model_call_count,
                tool_call_count=tool_call_count,
            )

    def run(
        self,
        user_message: str,
        *,
        context: ToolContext,
        conversation: Conversation = _EMPTY_CONVERSATION,
    ) -> AgentResult:
        with logfire.span("agent.run") as agent_span:
            loop_result = self._execute_agent_loop(
                user_message,
                context=context,
                conversation=conversation,
            )
            value = loop_result.value

            if not isinstance(value, Exception):
                agent_span.set_attributes(
                    {
                        "customer_support_agent.agent.outcome": "success",
                        "customer_support_agent.agent.model_call.count": loop_result.model_call_count,
                        "customer_support_agent.agent.tool_call.count": loop_result.tool_call_count,
                    }
                )
                return value

            error = value

            if isinstance(error, AgentError):
                agent_span.set_attributes(
                    {
                        "customer_support_agent.agent.outcome": "error",
                        "customer_support_agent.agent.error.code": error.code,
                        "customer_support_agent.agent.model_call.count": loop_result.model_call_count,
                        "customer_support_agent.agent.tool_call.count": loop_result.tool_call_count,
                    }
                )
            else:
                agent_span.set_attributes(
                    {
                        "customer_support_agent.agent.outcome": "exception",
                        "customer_support_agent.agent.model_call.count": loop_result.model_call_count,
                        "customer_support_agent.agent.tool_call.count": loop_result.tool_call_count,
                        "error.type": type(error).__name__,
                    }
                )

            agent_span.set_level("error")

        raise error
