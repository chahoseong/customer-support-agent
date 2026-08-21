import json
import os

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionUserMessageParam,
)

TOOL_NAME = "tool_calling_probe"
EXPECTED_MESSAGE = "tool-calling-ok"


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"{name} environment variable is required")

    return value.strip()


def main() -> int:
    # Configure the SDK client for the model server selected at runtime.
    base_url = get_required_env("LLM_BASE_URL")
    model_name = get_required_env("LLM_MODEL_NAME")
    api_key = os.getenv("LLM_API_KEY", "").strip() or "no-api-key"

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Define the deterministic tool contract and prompt used by the compatibility check.
    tools: list[ChatCompletionFunctionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": (
                    "A probe tool used to test whether structured tool calling "
                    "works through the configured API."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": f"Always set this value to {EXPECTED_MESSAGE!r}.",
                        },
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    messages: list[ChatCompletionUserMessageParam] = [
        {
            "role": "user",
            "content": (
                f"Call the {TOOL_NAME} tool with message set to {EXPECTED_MESSAGE!r}."
            ),
        }
    ]

    # Force the named tool call to isolate protocol compatibility from tool selection.
    tool_choice: ChatCompletionNamedToolChoiceParam = {
        "type": "function",
        "function": {"name": TOOL_NAME},
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )

    # Validate the OpenAI-compatible response shape and selected tool.
    tool_calls = response.choices[0].message.tool_calls or []

    if len(tool_calls) != 1:
        raise RuntimeError(f"Expected exactly one tool call, got {len(tool_calls)}")

    tool_call = tool_calls[0]

    if tool_call.type != "function":
        raise RuntimeError(
            f"Expected tool call type 'function', got {tool_call.type!r}"
        )

    if tool_call.function.name != TOOL_NAME:
        raise RuntimeError(
            f"Expected tool name {TOOL_NAME!r}, got {tool_call.function.name!r}"
        )

    # Decode and verify the tool arguments after the structural checks pass.
    raw_arguments = tool_call.function.arguments

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Tool arguments are not valid JSON: {raw_arguments!r}"
        ) from exc

    expected_arguments = {"message": EXPECTED_MESSAGE}

    if arguments != expected_arguments:
        raise RuntimeError(
            f"Expected arguments {expected_arguments!r}, got {arguments!r}"
        )

    print(f"PASS: model={model_name!r}, tool={TOOL_NAME!r}, arguments={arguments!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
