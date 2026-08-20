from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: object


@dataclass
class UserPromptPart:
    content: str


@dataclass
class ToolResultPart:
    tool_call_id: str
    result: object


type ModelRequestPart = UserPromptPart | ToolResultPart


@dataclass
class ModelRequest:
    parts: tuple[ModelRequestPart, ...]


@dataclass
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


type ModelMessage = ModelRequest | ModelResponse
