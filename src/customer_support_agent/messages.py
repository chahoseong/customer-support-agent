from dataclasses import dataclass

from pydantic import BaseModel


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
class ToolCallPart:
    id: str
    name: str
    arguments: object


@dataclass
class StructuredOutputPart:
    output: BaseModel


type ModelResponsePart = ToolCallPart | StructuredOutputPart


@dataclass
class ModelResponse:
    parts: tuple[ModelResponsePart, ...] = ()


type ModelMessage = ModelRequest | ModelResponse
