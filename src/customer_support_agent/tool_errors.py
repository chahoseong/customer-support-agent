from typing import Literal, TypedDict

type ToolErrorCode = Literal[
    "invalid_arguments",
    "order_not_found",
    "tool_execution_failed",
    "unknown_tool",
]


class ToolErrorDetails(TypedDict):
    code: ToolErrorCode
    message: str


class ToolError(TypedDict):
    error: ToolErrorDetails


_MESSAGES: dict[ToolErrorCode, str] = {
    "invalid_arguments": "Arguments do not match the tool's input schema.",
    "order_not_found": "No order matched the provided order_id.",
    "tool_execution_failed": ("The tool failed unexpectedly; do not assume a result."),
    "unknown_tool": (
        "The requested tool is not available. Use an available tool instead."
    ),
}


def create_tool_error(code: ToolErrorCode) -> ToolError:
    return {
        "error": {
            "code": code,
            "message": _MESSAGES[code],
        }
    }
