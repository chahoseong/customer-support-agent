from dataclasses import FrozenInstanceError

import pytest

from customer_support_agent.conversation import (
    AgentMessage,
    Conversation,
    UserMessage,
)


@pytest.mark.parametrize(
    ("value", "field_name", "replacement"),
    [
        pytest.param(
            UserMessage(content="user message"),
            "content",
            "replace message",
            id="user-message",
        ),
        pytest.param(
            AgentMessage(content="agent message"),
            "content",
            "replace message",
            id="agent-message",
        ),
        pytest.param(
            Conversation(),
            "messages",
            (UserMessage(content="user"), AgentMessage(content="agent")),
            id="conversation",
        ),
    ],
)
def test_conversation_values_reject_field_reassignment(
    value: object,
    field_name: str,
    replacement: object,
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, replacement)
