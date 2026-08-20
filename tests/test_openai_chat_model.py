from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletion

from customer_support_agent.messages import (
    ModelRequest,
    ModelResponse,
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
