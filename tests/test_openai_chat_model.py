from unittest.mock import Mock

import pytest
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, ConfigDict, ValidationError

from customer_support_agent.messages import (
    ModelRequest,
    ModelResponse,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models.openai import OpenAIChatModel
from customer_support_agent.tools.tool import ToolDefinition


class ExampleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str


EXAMPLE_TOOL_DEFINITION = ToolDefinition(
    name="example_tool",
    description="Return an example value.",
    parameters={
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
            }
        },
        "required": ["value"],
        "additionalProperties": False,
    },
)


def test_generate_converts_structured_output_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output = ExampleOutput(
        message="The example value was processed.",
    )

    sdk_response = ParsedChatCompletion[ExampleOutput].model_validate(
        {
            "id": "completion-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": expected_output.model_dump_json(),
                        "role": "assistant",
                        "parsed": expected_output,
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )

    client = Mock()
    client.chat.completions.parse.return_value = sdk_response

    openai_constructor = Mock(return_value=client)
    monkeypatch.setattr(
        "customer_support_agent.models.openai.OpenAI",
        openai_constructor,
    )

    model = OpenAIChatModel(
        base_url="http://model-server.test/v1",
        model_name="test-model",
        api_key="test-api-key",
    )

    response = model.generate(
        (ModelRequest(parts=(UserPromptPart(content="Use the example tool."),)),),
        (EXAMPLE_TOOL_DEFINITION,),
        output_type=ExampleOutput,
    )

    openai_constructor.assert_called_once_with(
        base_url="http://model-server.test/v1",
        api_key="test-api-key",
    )

    client.chat.completions.parse.assert_called_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Use the example tool.",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": EXAMPLE_TOOL_DEFINITION.name,
                    "description": EXAMPLE_TOOL_DEFINITION.description,
                    "parameters": EXAMPLE_TOOL_DEFINITION.parameters,
                    "strict": True,
                },
            }
        ],
        response_format=ExampleOutput,
        parallel_tool_calls=False,
    )

    assert response == ModelResponse(
        parts=(StructuredOutputPart(output=expected_output),)
    )


def test_generate_converts_tool_call_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_response = ParsedChatCompletion[ExampleOutput].model_validate(
        {
            "id": "completion-1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": EXAMPLE_TOOL_DEFINITION.name,
                                    "arguments": '{"value":"expected"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )

    client = Mock()
    client.chat.completions.parse.return_value = sdk_response

    monkeypatch.setattr(
        "customer_support_agent.models.openai.OpenAI",
        Mock(return_value=client),
    )

    model = OpenAIChatModel(
        base_url="http://model-server.test/v1",
        model_name="test-model",
        api_key="test-api-key",
    )

    response = model.generate(
        (ModelRequest(parts=(UserPromptPart(content="Use the example tool."),)),),
        (EXAMPLE_TOOL_DEFINITION,),
        output_type=ExampleOutput,
    )

    assert response == ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name=EXAMPLE_TOOL_DEFINITION.name,
                arguments={"value": "expected"},
            ),
        )
    )


def test_generate_converts_tool_call_history_and_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output = ExampleOutput(
        message="The example value was processed.",
    )

    sdk_response = ParsedChatCompletion[ExampleOutput].model_validate(
        {
            "id": "completion-2",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": expected_output.model_dump_json(),
                        "role": "assistant",
                        "parsed": expected_output,
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )
    client = Mock()
    client.chat.completions.parse.return_value = sdk_response

    monkeypatch.setattr(
        "customer_support_agent.models.openai.OpenAI",
        Mock(return_value=client),
    )

    model = OpenAIChatModel(
        base_url="http://model-server.test/v1",
        model_name="test-model",
        api_key="test-api-key",
    )

    model.generate(
        (
            ModelRequest(parts=(UserPromptPart(content="Use the example tool."),)),
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=EXAMPLE_TOOL_DEFINITION.name,
                        arguments={"value": "expected"},
                    ),
                )
            ),
            ModelRequest(
                parts=(
                    ToolResultPart(
                        tool_call_id="call-1",
                        result={"value": "expected"},
                    ),
                )
            ),
        ),
        (EXAMPLE_TOOL_DEFINITION,),
        output_type=ExampleOutput,
    )

    sent_messages = client.chat.completions.parse.call_args.kwargs["messages"]
    assert sent_messages == [
        {
            "role": "user",
            "content": "Use the example tool.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": EXAMPLE_TOOL_DEFINITION.name,
                        "arguments": '{"value": "expected"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"value": "expected"}',
        },
    ]


def test_generate_sends_instructions_as_system_message_before_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output = ExampleOutput(
        message="The example value was processed.",
    )

    sdk_response = ParsedChatCompletion[ExampleOutput].model_validate(
        {
            "id": "completion-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": expected_output.model_dump_json(),
                        "role": "assistant",
                        "parsed": expected_output,
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )

    client = Mock()
    client.chat.completions.parse.return_value = sdk_response

    monkeypatch.setattr(
        "customer_support_agent.models.openai.OpenAI",
        Mock(return_value=client),
    )

    model = OpenAIChatModel(
        base_url="http://model-server.test/v1",
        model_name="test-model",
        api_key="test-api-key",
    )

    instructions = "Follow the customer support instructions."

    model.generate(
        (ModelRequest(parts=(UserPromptPart(content="Where is my order?"),)),),
        (),
        output_type=ExampleOutput,
        instructions=instructions,
    )

    sent_messages = client.chat.completions.parse.call_args.kwargs["messages"]

    assert sent_messages == [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": "Where is my order?",
        },
    ]


def test_generate_returns_response_without_output_when_model_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_response = ParsedChatCompletion[ExampleOutput].model_validate(
        {
            "id": "completion-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "parsed": None,
                        "refusal": "I cannot complete this request.",
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )

    client = Mock()
    client.chat.completions.parse.return_value = sdk_response

    monkeypatch.setattr(
        "customer_support_agent.models.openai.OpenAI",
        Mock(return_value=client),
    )

    model = OpenAIChatModel(
        base_url="http://model-server.test/v1",
        model_name="test-model",
        api_key="test-api-key",
    )

    response = model.generate(
        (ModelRequest(parts=(UserPromptPart(content="Where is my order?"),)),),
        (),
        output_type=ExampleOutput,
    )

    assert response == ModelResponse()


def test_generate_returns_response_without_output_when_structured_output_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ExampleOutput.model_validate({})

    client = Mock()
    client.chat.completions.parse.side_effect = exc_info.value

    monkeypatch.setattr(
        "customer_support_agent.models.openai.OpenAI",
        Mock(return_value=client),
    )

    model = OpenAIChatModel(
        base_url="http://model-server.test/v1",
        model_name="test-model",
        api_key="test-api-key",
    )

    response = model.generate(
        (ModelRequest(parts=(UserPromptPart(content="Where is my order?"),)),),
        (),
        output_type=ExampleOutput,
    )

    assert response == ModelResponse()


def test_generate_converts_text_response_history_to_assistant_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output = ExampleOutput(
        message="Order order-002 has shipped.",
    )

    sdk_response = ParsedChatCompletion[ExampleOutput].model_validate(
        {
            "id": "completion-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": expected_output.model_dump_json(),
                        "role": "assistant",
                        "parsed": expected_output,
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )

    client = Mock()
    client.chat.completions.parse.return_value = sdk_response

    monkeypatch.setattr(
        "customer_support_agent.models.openai.OpenAI",
        Mock(return_value=client),
    )

    model = OpenAIChatModel(
        base_url="http://model-server.test/v1",
        model_name="test-model",
        api_key="test-api-key",
    )

    model.generate(
        (
            ModelRequest(
                parts=(
                    UserPromptPart(
                        content="Where is my shipped order?",
                    ),
                )
            ),
            ModelResponse(
                parts=(
                    TextPart(
                        content="Please provide the order ID.",
                    ),
                )
            ),
            ModelRequest(
                parts=(
                    UserPromptPart(
                        content="order-002",
                    ),
                )
            ),
        ),
        (),
        output_type=ExampleOutput,
    )

    sent_messages = client.chat.completions.parse.call_args.kwargs["messages"]

    assert sent_messages == [
        {
            "role": "user",
            "content": "Where is my shipped order?",
        },
        {
            "role": "assistant",
            "content": "Please provide the order ID.",
        },
        {
            "role": "user",
            "content": "order-002",
        },
    ]
