from collections.abc import Sequence

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)

from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from customer_support_agent.models.base import ChatModel
from customer_support_agent.tools.definitions import ToolDefinition


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
    ) -> ModelResponse:
        sdk_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            if not isinstance(message, ModelRequest):
                raise NotImplementedError("Model responses are not supported yet.")

            for part in message.parts:
                if not isinstance(part, UserPromptPart):
                    raise NotImplementedError("Tool results are not supported yet.")

                sdk_messages.append(
                    {
                        "role": "user",
                        "content": part.content,
                    }
                )

        sdk_tools: list[ChatCompletionFunctionToolParam] = [
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

        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=sdk_messages,
            tools=sdk_tools,
            parallel_tool_calls=False,
        )

        return ModelResponse(
            content=completion.choices[0].message.content,
        )
