import pytest
from evals.order.capture import ToolUseRecordingModel
from evals.order.models import (
    ObservedToolUse,
    OrderEvalOutput,
)

from customer_support_agent.agent import AgentResult
from customer_support_agent.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UserPromptPart,
)
from customer_support_agent.tools import ToolDefinition
from customer_support_agent.tools.errors import (
    ToolErrorCode,
    create_tool_error,
)
from tests.scripted_model import ScriptedModel


def test_tool_use_recording_model_records_tool_use_after_corresponding_result() -> None:
    initial_request = ModelRequest(
        parts=(UserPromptPart(content="Where is order-002?"),)
    )

    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name="find_order",
                arguments={"order_id": "order-002"},
            ),
        )
    )

    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={"order_id": "order-002", "status": "shipped"},
            ),
        )
    )

    final_response = ModelResponse(
        parts=(TextPart(content="Order order-002 has shipped."),)
    )

    inner_model = ScriptedModel(responses=[tool_call_response, final_response])
    recording_model = ToolUseRecordingModel(inner_model)

    recording_model.generate((initial_request,), ())
    recording_model.generate(
        (
            initial_request,
            tool_call_response,
            tool_result_request,
        ),
        (),
    )

    assert recording_model.tool_uses == (
        ObservedToolUse(
            tool_name="find_order",
            arguments={"order_id": "order-002"},
            outcome="success",
        ),
    )


def test_tool_use_recording_model_returns_inner_model_response() -> None:
    expected_response = ModelResponse(
        parts=(TextPart(content="Expected response"),),
    )
    inner_model = ScriptedModel(responses=[expected_response])
    recording_model = ToolUseRecordingModel(inner_model)

    response = recording_model.generate((), ())

    assert response == expected_response


def test_tool_use_recording_model_records_result_once_when_history_repeats() -> None:
    initial_request = ModelRequest(
        parts=(UserPromptPart(content="Where is order-002?"),),
    )
    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name="find_order",
                arguments={"order_id": "order-002"},
            ),
        ),
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={"order_id": "order-002", "status": "shipped"},
            ),
        ),
    )
    repeated_history = (
        initial_request,
        tool_call_response,
        tool_result_request,
    )
    inner_model = ScriptedModel(
        responses=[
            tool_call_response,
            ModelResponse(),
            ModelResponse(),
        ],
    )
    recording_model = ToolUseRecordingModel(inner_model)

    recording_model.generate((initial_request,), ())
    recording_model.generate(repeated_history, ())
    recording_model.generate(repeated_history, ())

    assert len(recording_model.tool_uses) == 1


def test_tool_use_recording_model_preserves_tool_call_order_across_requests() -> None:
    initial_request = ModelRequest(
        parts=(UserPromptPart(content="Where is order-002?"),),
    )
    order_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name="find_order",
                arguments={"order_id": "order-002"},
            ),
        ),
    )
    order_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={"order_id": "order-002", "status": "shipped"},
            ),
        ),
    )
    shipment_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-2",
                name="find_shipment",
                arguments={"order_id": "order-002"},
            ),
        ),
    )
    shipment_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-2",
                result={
                    "order_id": "order-002",
                    "shipment_status": "in_transit",
                },
            ),
        ),
    )
    final_response = ModelResponse(
        parts=(TextPart(content="The shipment is in transit."),),
    )
    inner_model = ScriptedModel(
        responses=[
            order_call_response,
            shipment_call_response,
            final_response,
        ],
    )
    recording_model = ToolUseRecordingModel(inner_model)

    recording_model.generate((initial_request,), ())
    recording_model.generate(
        (
            initial_request,
            order_call_response,
            order_result_request,
        ),
        (),
    )
    recording_model.generate(
        (
            initial_request,
            order_call_response,
            order_result_request,
            shipment_call_response,
            shipment_result_request,
        ),
        (),
    )

    assert tuple(tool_use.tool_name for tool_use in recording_model.tool_uses) == (
        "find_order",
        "find_shipment",
    )


@pytest.mark.parametrize(
    "error_code",
    [
        pytest.param("invalid_arguments", id="invalid-arguments"),
        pytest.param("order_not_found", id="order-not-found"),
        pytest.param("shipment_not_found", id="shipment-not-found"),
        pytest.param("tool_execution_failed", id="tool-execution-failed"),
        pytest.param("unknown_tool", id="unknown-tool"),
    ],
)
def test_tool_use_recording_model_records_structured_tool_error_outcome(
    error_code: ToolErrorCode,
) -> None:
    initial_request = ModelRequest(
        parts=(UserPromptPart(content="Look up the order."),),
    )
    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name="find_order",
                arguments={"order_id": "order-002"},
            ),
        ),
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result=create_tool_error(error_code),
            ),
        ),
    )
    inner_model = ScriptedModel(
        responses=[
            tool_call_response,
            ModelResponse(),
        ],
    )
    recording_model = ToolUseRecordingModel(inner_model)

    recording_model.generate((initial_request,), ())
    recording_model.generate(
        (
            initial_request,
            tool_call_response,
            tool_result_request,
        ),
        (),
    )

    assert recording_model.tool_uses[0].outcome == error_code


def test_tool_use_recording_model_records_uninterpretable_outcome_for_invalid_tool_error() -> (
    None
):
    initial_request = ModelRequest(
        parts=(UserPromptPart(content="Look up the order."),),
    )
    tool_call_response = ModelResponse(
        parts=(
            ToolCallPart(
                id="call-1",
                name="find_order",
                arguments={"order_id": "order-002"},
            ),
        ),
    )
    tool_result_request = ModelRequest(
        parts=(
            ToolResultPart(
                tool_call_id="call-1",
                result={
                    "error": {
                        "code": "unsupported_failure",
                        "message": "Unsupported failure.",
                    }
                },
            ),
        ),
    )
    inner_model = ScriptedModel(
        responses=[
            tool_call_response,
            ModelResponse(),
        ],
    )
    recording_model = ToolUseRecordingModel(inner_model)

    recording_model.generate((initial_request,), ())
    recording_model.generate(
        (
            initial_request,
            tool_call_response,
            tool_result_request,
        ),
        (),
    )

    assert recording_model.tool_uses[0].outcome == "uninterpretable"


def test_tool_use_recording_model_delegates_generation_inputs_to_inner_model() -> None:
    request = ModelRequest(
        parts=(UserPromptPart(content="Look up the order."),),
    )
    tool_definition = ToolDefinition(
        name="find_order",
        description="Find an order.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
            },
        },
    )
    expected_response = ModelResponse(
        parts=(TextPart(content="Expected response"),),
    )
    inner_model = ScriptedModel(responses=[expected_response])
    recording_model = ToolUseRecordingModel(inner_model)

    recording_model.generate(
        [request],
        [tool_definition],
        output_type=AgentResult,
        instructions="Use tools when required.",
    )

    assert (
        inner_model.requests,
        inner_model.received_tools,
        inner_model.received_output_types,
        inner_model.received_instructions,
    ) == (
        [(request,)],
        [(tool_definition,)],
        [AgentResult],
        ["Use tools when required."],
    )


def test_order_eval_output_preserves_agent_result_and_tool_uses() -> None:
    agent_result = AgentResult(
        message="Order order-002 has shipped.",
    )
    tool_use = ObservedToolUse(
        tool_name="find_order",
        arguments={"order_id": "order-002"},
        outcome="success",
    )

    output = OrderEvalOutput(
        agent_result=agent_result,
        tool_uses=(tool_use,),
    )

    assert (
        output.agent_result,
        output.tool_uses,
    ) == (
        agent_result,
        (tool_use,),
    )
