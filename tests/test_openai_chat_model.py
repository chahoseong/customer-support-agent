from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion

from customer_support_agent.messages import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.models.openai import OpenAIChatModel
from customer_support_agent.tools.get_order import GET_ORDER_TOOL_DEFINITION


def test_generate_converts_text_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_response = ChatCompletion.model_validate(
        {
            "id": "completion-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Your order is being processed.",
                        "role": "assistant",
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )

    client = Mock()
    client.chat.completions.create.return_value = sdk_response
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
        (ModelRequest(parts=(UserPromptPart(content="Where is order-001?"),)),),
        (GET_ORDER_TOOL_DEFINITION,),
    )

    openai_constructor.assert_called_once_with(
        base_url="http://model-server.test/v1",
        api_key="test-api-key",
    )
    client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Where is order-001?",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": GET_ORDER_TOOL_DEFINITION.name,
                    "description": GET_ORDER_TOOL_DEFINITION.description,
                    "parameters": GET_ORDER_TOOL_DEFINITION.parameters,
                },
            }
        ],
        parallel_tool_calls=False,
    )

    assert response == ModelResponse(content="Your order is being processed.")


def test_generate_converts_tool_call_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_response = ChatCompletion.model_validate(
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
                                    "name": "get_order",
                                    "arguments": '{"order_id":"order-001"}',
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
    client.chat.completions.create.return_value = sdk_response

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
        (ModelRequest(parts=(UserPromptPart(content="Where is order-001?"),)),),
        (GET_ORDER_TOOL_DEFINITION,),
    )

    assert response == ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name="get_order",
                arguments={"order_id": "order-001"},
            ),
        )
    )


def test_generate_converts_tool_call_history_and_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_response = ChatCompletion.model_validate(
        {
            "id": "completion-2",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Your order is being processed.",
                        "role": "assistant",
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )
    client = Mock()
    client.chat.completions.create.return_value = sdk_response
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
            ModelRequest(parts=(UserPromptPart(content="Where is order-001?"),)),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="get_order",
                        arguments={"order_id": "order-001"},
                    ),
                )
            ),
            ModelRequest(
                parts=(
                    ToolResultPart(
                        tool_call_id="call-1",
                        result={
                            "order_id": "order-001",
                            "status": "processing",
                        },
                    ),
                )
            ),
        ),
        (GET_ORDER_TOOL_DEFINITION,),
    )

    sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages == [
        {
            "role": "user",
            "content": "Where is order-001?",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": "order-001"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": ('{"order_id": "order-001", "status": "processing"}'),
        },
    ]
