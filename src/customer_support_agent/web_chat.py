import logging
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

import gradio as gr

from customer_support_agent.agent import Agent, AgentError
from customer_support_agent.conversation import (
    AgentMessage,
    Conversation,
    UserMessage,
)
from customer_support_agent.tools import ToolContext

logger = logging.getLogger(__name__)

_CUSTOMER_CONTEXT = ToolContext(customer_id="customer-001")
_CHAT_ERROR_MESSAGE = "The request could not be completed. Please try again."


class ChatTextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ChatHistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: list[ChatTextBlock]


type ChatResponse = str | list[ChatHistoryMessage]
type ChatCallback = Callable[[str, list[ChatHistoryMessage]], ChatResponse]


def conversation_from_chat_history(
    history: Sequence[ChatHistoryMessage],
) -> Conversation:
    messages: list[UserMessage | AgentMessage] = []

    for history_message in history:
        content = "".join(block["text"] for block in history_message["content"])

        if history_message["role"] == "user":
            messages.append(UserMessage(content=content))
        else:
            messages.append(AgentMessage(content=content))

    return Conversation(messages=tuple(messages))


def create_chat_callback(agent: Agent) -> ChatCallback:
    def respond(message: str, history: list[ChatHistoryMessage]) -> ChatResponse:
        conversation = conversation_from_chat_history(history)

        try:
            result = agent.run(
                message,
                context=_CUSTOMER_CONTEXT,
                conversation=conversation,
            )
        except AgentError:
            logger.exception(
                "The customer support agent could not complete the request."
            )
            gr.Warning(_CHAT_ERROR_MESSAGE, title="Error")
            return []

        return result.message

    return respond


def create_chat_interface(agent: Agent) -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=create_chat_callback(agent),
        title="Customer Support Agent",
        flagging_mode="never",
        concurrency_limit=1,
        show_progress="full",
        save_history=False,
        api_visibility="private",
    )
