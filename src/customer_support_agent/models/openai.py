import json
from collections.abc import Sequence

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)

from customer_support_agent.messages import (
    ModelMessage,
    ModelResponse,
    ToolCall,
    UserPromptPart,
)
from customer_support_agent.models.base import ChatModel
from customer_support_agent.tools import ToolDefinition


def _to_sdk_messages(
    messages: Sequence[ModelMessage],
    *,
    instructions: str | None = None,
) -> list[ChatCompletionMessageParam]:
    sdk_messages: list[ChatCompletionMessageParam] = []

    if instructions is not None:
        sdk_messages.append(
            {
                "role": "system",
                "content": instructions,
            }
        )

    for message in messages:
        if isinstance(message, ModelResponse):
            sdk_tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in message.tool_calls
            ]
            sdk_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": sdk_tool_calls,
                }
            )
            continue

        for part in message.parts:
            if isinstance(part, UserPromptPart):
                sdk_messages.append(
                    {
                        "role": "user",
                        "content": part.content,
                    }
                )
            else:
                sdk_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": json.dumps(part.result),
                    }
                )

    return sdk_messages


def _to_sdk_tools(
    tools: Sequence[ToolDefinition],
) -> list[ChatCompletionFunctionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _to_model_response(
    response_message: ChatCompletionMessage,
) -> ModelResponse:
    tool_calls: list[ToolCall] = []

    for sdk_tool_call in response_message.tool_calls or []:
        if sdk_tool_call.type != "function":
            raise NotImplementedError("Custom tool calls are not supported.")

        tool_calls.append(
            ToolCall(
                id=sdk_tool_call.id,
                name=sdk_tool_call.function.name,
                arguments=json.loads(sdk_tool_call.function.arguments),
            )
        )

    return ModelResponse(
        content=response_message.content,
        tool_calls=tuple(tool_calls),
    )


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._model_name = model_name

    def generate(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        *,
        instructions: str | None = None,
    ) -> ModelResponse:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=_to_sdk_messages(messages, instructions=instructions),
            tools=_to_sdk_tools(tools),
            parallel_tool_calls=False,
        )

        return _to_model_response(completion.choices[0].message)
