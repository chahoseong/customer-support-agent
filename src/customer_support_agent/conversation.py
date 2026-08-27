from dataclasses import dataclass


@dataclass(frozen=True)
class UserMessage:
    content: str


@dataclass(frozen=True)
class AgentMessage:
    content: str


@dataclass(frozen=True)
class Conversation:
    messages: tuple[UserMessage | AgentMessage, ...] = ()
