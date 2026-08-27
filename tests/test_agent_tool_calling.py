from customer_support_agent.agent import Agent, AgentResult
from customer_support_agent.messages import (
    ModelRequest,
    ModelResponse,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.tools.tool import ToolContext, tool
from customer_support_agent.tools.toolset import Toolset
from tests.scripted_model import ScriptedModel

EMPTY_TOOLSET = Toolset(tools=())
TEST_CONTEXT = ToolContext(customer_id="customer-001")


def test_agent_sends_tool_result_to_model_and_returns_agent_result() -> None:
    @tool
    def configured_tool(
        context: ToolContext,
        value: str,
    ) -> dict[str, str]:
        """Return the configured value with its customer."""
        return {
            "customer_id": context.customer_id,
            "value": value,
        }

    toolset = Toolset(tools=(configured_tool,))
    context = ToolContext(customer_id="customer-001")

    user_request = ModelRequest(
        parts=(UserPromptPart(content="Use the configured tool."),)
    )
    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name=configured_tool.definition.name,
                arguments={"value": "expected"},
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "customer_id": "customer-001",
                    "value": "expected",
                },
            ),
        )
    )

    expected_result = AgentResult(
        message="The configured value is expected.",
    )

    final_response = ModelResponse(
        parts=(StructuredOutputPart(output=expected_result),)
    )
    final_text_response = ModelResponse(
        parts=(TextPart(content=expected_result.message),)
    )

    model = ScriptedModel(
        [
            tool_call_response,
            final_text_response,
            final_response,
        ]
    )

    result = Agent(model, toolset).run(
        "Use the configured tool.",
        context=context,
    )

    assert result == expected_result
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_preserves_message_history_across_rounds_with_different_tools() -> None:
    @tool
    def first_tool(
        context: ToolContext,
        first_value: str,
    ) -> dict[str, str]:
        """Return the first value with its customer."""
        return {
            "customer_id": context.customer_id,
            "first_value": first_value,
        }

    @tool
    def second_tool(
        context: ToolContext,
        second_value: str,
    ) -> dict[str, str]:
        """Return the second value with its customer."""
        return {
            "customer_id": context.customer_id,
            "second_value": second_value,
        }

    toolset = Toolset(tools=(first_tool, second_tool))
    context = ToolContext(customer_id="customer-001")

    user_request = ModelRequest(
        parts=(
            UserPromptPart(
                content="Process the first and second values.",
            ),
        )
    )
    first_tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name=first_tool.definition.name,
                arguments={"first_value": "first"},
            ),
        )
    )
    first_tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "customer_id": "customer-001",
                    "first_value": "first",
                },
            ),
        )
    )
    second_tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-2",
                name=second_tool.definition.name,
                arguments={"second_value": "second"},
            ),
        )
    )
    second_tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-2",
                result={
                    "customer_id": "customer-001",
                    "second_value": "second",
                },
            ),
        )
    )

    expected_result = AgentResult(
        message="The first and second values were processed.",
    )
    final_response = ModelResponse(
        parts=(StructuredOutputPart(output=expected_result),)
    )
    final_text_response = ModelResponse(
        parts=(TextPart(content=expected_result.message),)
    )

    model = ScriptedModel(
        [
            first_tool_call_response,
            second_tool_call_response,
            final_text_response,
            final_response,
        ]
    )

    result = Agent(model, toolset).run(
        "Process the first and second values.", context=context
    )

    assert result == expected_result
    assert model.requests == [
        (user_request,),
        (
            user_request,
            first_tool_call_response,
            first_tool_result_request,
        ),
        (
            user_request,
            first_tool_call_response,
            first_tool_result_request,
            second_tool_call_response,
            second_tool_result_request,
        ),
        (
            user_request,
            first_tool_call_response,
            first_tool_result_request,
            second_tool_call_response,
            second_tool_result_request,
        ),
    ]


def test_agent_prioritizes_tool_calls_when_response_also_contains_agent_result() -> (
    None
):
    @tool
    def configured_tool(
        context: ToolContext,
        value: str,
    ) -> dict[str, str]:
        """Return the configured value with its customer."""
        return {
            "customer_id": context.customer_id,
            "value": value,
        }

    toolset = Toolset(tools=(configured_tool,))

    user_message = "Process the configured value."
    user_request = ModelRequest(parts=(UserPromptPart(content=user_message),))

    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name=configured_tool.definition.name,
                arguments={"value": "expected"},
            ),
            StructuredOutputPart(
                output=AgentResult(
                    message="The configured value is available.",
                )
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "customer_id": "customer-001",
                    "value": "expected",
                },
            ),
        )
    )

    expected_result = AgentResult(
        message="The configured value is expected.",
    )

    final_response = ModelResponse(
        parts=(StructuredOutputPart(output=expected_result),)
    )
    final_text_response = ModelResponse(
        parts=(TextPart(content=expected_result.message),)
    )

    model = ScriptedModel(
        [
            tool_call_response,
            final_text_response,
            final_response,
        ]
    )

    result = Agent(model, toolset).run(user_message, context=TEST_CONTEXT)

    assert result == expected_result
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_returns_unknown_tool_error_to_model_and_continues() -> None:
    user_message = "Use an unavailable tool."
    user_request = ModelRequest(parts=(UserPromptPart(content=user_message),))

    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name="unknown_tool",
                arguments={},
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "error": {
                        "code": "unknown_tool",
                        "message": (
                            "The requested tool is not available. "
                            "Use an available tool instead."
                        ),
                    }
                },
            ),
        )
    )

    expected_result = AgentResult(
        message="I cannot use that tool.",
    )

    model = ScriptedModel(
        [
            tool_call_response,
            ModelResponse(parts=(TextPart(content=expected_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    result = Agent(model, EMPTY_TOOLSET).run(user_message, context=TEST_CONTEXT)

    assert result == expected_result
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_returns_tool_execution_failed_error_to_model_and_continues() -> None:
    @tool
    def broken_tool(
        value: str,
    ) -> object:
        """Raise runtime error"""
        raise RuntimeError("sensitive internal failure")

    toolset = Toolset(tools=(broken_tool,))

    user_message = "Complete the configured action."
    user_request = ModelRequest(parts=(UserPromptPart(content=user_message),))

    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name=broken_tool.definition.name,
                arguments={"value": "expected"},
            ),
        )
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "error": {
                        "code": "tool_execution_failed",
                        "message": (
                            "The tool failed unexpectedly; do not assume a result."
                        ),
                    }
                },
            ),
        )
    )

    expected_result = AgentResult(
        message="I could not complete the action.",
    )

    model = ScriptedModel(
        [
            tool_call_response,
            ModelResponse(parts=(TextPart(content=expected_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    result = Agent(model, toolset).run(user_message, context=TEST_CONTEXT)

    assert result == expected_result
    assert model.requests == [
        (user_request,),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
        (
            user_request,
            tool_call_response,
            tool_result_request,
        ),
    ]


def test_agent_omits_tool_definitions_from_final_model_call() -> None:
    @tool
    def configured_tool(
        context: ToolContext,
        value: str,
    ) -> str:
        """Return the provided example value."""
        return value

    toolset = Toolset(tools=(configured_tool,))
    context = ToolContext(customer_id="customer-001")

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=configured_tool.definition.name,
                        arguments={"value": "expected"},
                    ),
                )
            ),
            ModelResponse(parts=(TextPart(content="The order is processing."),)),
            ModelResponse(
                parts=(
                    StructuredOutputPart(
                        output=AgentResult(
                            message="The order is processing.",
                        )
                    ),
                )
            ),
        ]
    )

    Agent(model, toolset).run(
        "Use the configured tool.",
        context=context,
    )

    assert model.received_tools == [
        toolset.definitions,
        toolset.definitions,
        (),
    ]
