from unittest.mock import Mock

import gradio as gr
import pytest

from customer_support_agent.agent import Agent, AgentError, AgentResult
from customer_support_agent.conversation import (
    AgentMessage,
    Conversation,
    UserMessage,
)
from customer_support_agent.tools import ToolContext
from customer_support_agent.web_chat import (
    ChatHistoryMessage,
    conversation_from_chat_history,
    create_chat_callback,
    create_chat_interface,
)


def test_chat_history_preserves_message_roles_and_order() -> None:
    history: list[ChatHistoryMessage] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Where is my order?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Which order do you mean?"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "order-002"}],
        },
    ]

    conversation = conversation_from_chat_history(history)

    assert conversation == Conversation(
        messages=(
            UserMessage(content="Where is my order?"),
            AgentMessage(content="Which order do you mean?"),
            UserMessage(content="order-002"),
        )
    )


def test_chat_callback_passes_current_message_conversation_and_customer_context_to_agent() -> (
    None
):
    agent = Mock(spec=Agent)
    agent.run.return_value = AgentResult(message="Order order-002 has shipped.")
    history: list[ChatHistoryMessage] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Where is my shipped order?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Please provide the order ID."}],
        },
    ]

    callback = create_chat_callback(agent)

    callback("order-002", history)

    agent.run.assert_called_once_with(
        "order-002",
        context=ToolContext(customer_id="customer-001"),
        conversation=Conversation(
            messages=(
                UserMessage(content="Where is my shipped order?"),
                AgentMessage(content="Please provide the order ID."),
            )
        ),
    )


def test_chat_callback_returns_agent_result_message() -> None:
    agent = Mock(spec=Agent)
    agent.run.return_value = AgentResult(message="Order order-001 is processing.")

    callback = create_chat_callback(agent)

    response = callback("Where is order-001?", [])

    assert response == "Order order-001 is processing."


def test_chat_callback_reports_failure_outside_conversation_when_agent_run_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Mock(spec=Agent)
    agent.run.side_effect = AgentError("model_call_failed")
    warning = Mock()
    monkeypatch.setattr(gr, "Warning", warning)

    callback = create_chat_callback(agent)

    response = callback("Where is order-001?", [])

    assert response == []
    warning.assert_called_once_with(
        "The request could not be completed. Please try again.",
        title="Error",
    )


def test_chat_interface_uses_customer_support_callback_and_runtime_settings() -> None:
    agent = Mock(spec=Agent)
    agent.run.return_value = AgentResult(message="Order order-001 is processing.")

    interface = create_chat_interface(agent)

    assert isinstance(interface, gr.ChatInterface)
    assert interface.fn("Where is order-001?", []) == ("Order order-001 is processing.")
    assert interface.title == "Customer Support Agent"
    assert interface.concurrency_limit == 1
    assert interface.show_progress == "full"
    assert interface.save_history is False
    assert interface.flagging_mode == "never"
    assert interface.api_visibility == "private"
