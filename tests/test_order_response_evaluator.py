import asyncio
from datetime import UTC, datetime
from typing import Literal, cast

import pytest
from evals.order.models import (
    ExpectedToolUse,
    ObservedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
    ResponseCriterion,
)
from evals.order.response_evaluator import ResponseCriterionEvaluator
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import (
    EvaluationReason,
    EvaluatorContext,
)
from pydantic_evals.otel import SpanNode, SpanTree

from customer_support_agent.agent import AgentResult


def _create_response_evaluator_context(
    *,
    criterion: ResponseCriterion,
    criterion_kind: Literal["required", "forbidden"],
    response: str,
) -> EvaluatorContext[
    OrderEvalInput,
    OrderEvalOutput,
    OrderEvalMetadata,
]:
    return EvaluatorContext[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ](
        name=f"{criterion_kind}-response-criterion",
        inputs=OrderEvalInput(
            user_message="What is the status of order-003?",
            customer_id="customer-001",
            execution_condition="default",
        ),
        metadata=OrderEvalMetadata(
            scenario_id="scenario-1",
            required_tool_uses=(),
            forbidden_tools=frozenset(),
            required_tool_sequence=(),
            required_response_criteria=(
                (criterion,) if criterion_kind == "required" else ()
            ),
            forbidden_response_criteria=(
                (criterion,) if criterion_kind == "forbidden" else ()
            ),
        ),
        expected_output=None,
        output=OrderEvalOutput(
            agent_result=AgentResult(message=response),
            tool_uses=(),
        ),
        duration=0.0,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )


def _get_captured_judge_prompt(messages: list[ModelMessage]) -> str:
    judge_request = next(
        message for message in messages if isinstance(message, ModelRequest)
    )

    return "\n".join(
        part.content
        for part in judge_request.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def test_response_criterion_evaluator_returns_judge_assertion_and_reason_for_required_criterion() -> (
    None
):
    criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    context = _create_response_evaluator_context(
        criterion=criterion,
        criterion_kind="required",
        response="Order-003 has been delivered.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="required",
        judge_model=TestModel(
            custom_output_args={
                "reason": "The response states that the order was delivered.",
                "pass": True,
                "score": 1.0,
            }
        ),
    )

    result = asyncio.run(evaluator.evaluate(context))

    assert result == EvaluationReason(
        value=True,
        reason="The response states that the order was delivered.",
    )


def test_response_criterion_evaluator_returns_failed_assertion_when_required_criterion_is_missing() -> (
    None
):
    criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    context = _create_response_evaluator_context(
        criterion=criterion,
        criterion_kind="required",
        response="Order-003 is still processing.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="required",
        judge_model=TestModel(
            custom_output_args={
                "reason": "The response does not state that the order was delivered.",
                "pass": False,
                "score": 0.0,
            }
        ),
    )

    result = asyncio.run(evaluator.evaluate(context))

    assert result == EvaluationReason(
        value=False,
        reason="The response does not state that the order was delivered.",
    )


def test_response_criterion_evaluator_returns_failed_assertion_when_forbidden_criterion_is_present() -> (
    None
):
    criterion = ResponseCriterion(
        id="states_tracking_number",
        statement="The response states a tracking number.",
    )
    context = _create_response_evaluator_context(
        criterion=criterion,
        criterion_kind="forbidden",
        response="The tracking number is tracking-002.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="forbidden",
        judge_model=TestModel(
            custom_output_args={
                "reason": "The response includes a tracking number.",
                "pass": True,
                "score": 1.0,
            }
        ),
    )

    result = asyncio.run(evaluator.evaluate(context))

    assert result == EvaluationReason(
        value=False,
        reason="The response includes a tracking number.",
    )


def test_response_criterion_evaluator_returns_passed_assertion_when_forbidden_criterion_is_absent() -> (
    None
):
    criterion = ResponseCriterion(
        id="states_tracking_number",
        statement="The response states a tracking number.",
    )
    context = _create_response_evaluator_context(
        criterion=criterion,
        criterion_kind="forbidden",
        response="Order-002 is on its way.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="forbidden",
        judge_model=TestModel(
            custom_output_args={
                "reason": "The response does not include a tracking number.",
                "pass": False,
                "score": 0.0,
            }
        ),
    )

    result = asyncio.run(evaluator.evaluate(context))

    assert result == EvaluationReason(
        value=True,
        reason="The response does not include a tracking number.",
    )


def test_response_criterion_evaluator_returns_stable_name_for_required_criterion() -> (
    None
):
    criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="required",
        judge_model=TestModel(),
    )

    result = evaluator.get_default_evaluation_name()

    assert result == "response_required_states_order_status_delivered"


def test_response_criterion_evaluator_returns_stable_name_for_forbidden_criterion() -> (
    None
):
    criterion = ResponseCriterion(
        id="states_tracking_number",
        statement="The response states a tracking number.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="forbidden",
        judge_model=TestModel(),
    )

    result = evaluator.get_default_evaluation_name()

    assert result == "response_forbidden_states_tracking_number"


def test_response_criterion_evaluator_rejects_missing_judge_model() -> None:
    criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )

    with pytest.raises(
        ValueError,
        match=r"Judge model is required.",
    ):
        ResponseCriterionEvaluator(
            criterion=criterion,
            criterion_kind="required",
            judge_model=cast(Model, None),
        )


def test_response_criterion_evaluator_passes_model_settings_to_judge_model() -> None:
    received_model_settings: list[ModelSettings | None] = []

    def return_judge_result(
        _messages: list[ModelMessage],
        agent_info: AgentInfo,
    ) -> ModelResponse:
        received_model_settings.append(agent_info.model_settings)
        output_tool = agent_info.output_tools[0]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=output_tool.name,
                    args={
                        "reason": "The response states that the order was delivered.",
                        "pass": True,
                        "score": 1.0,
                    },
                )
            ]
        )

    criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    context = _create_response_evaluator_context(
        criterion=criterion,
        criterion_kind="required",
        response="Order-003 has been delivered.",
    )
    model_settings = ModelSettings(temperature=0.0)
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="required",
        judge_model=FunctionModel(return_judge_result),
        judge_model_settings=model_settings,
    )

    asyncio.run(evaluator.evaluate(context))

    assert received_model_settings == [model_settings]


def test_response_criterion_evaluator_sends_user_message_response_and_target_criterion_to_judge() -> (
    None
):
    target_criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    context = _create_response_evaluator_context(
        criterion=target_criterion,
        criterion_kind="required",
        response="Order-003 was delivered on January 15.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=target_criterion,
        criterion_kind="required",
        judge_model=TestModel(
            custom_output_args={
                "reason": "The response states that the order was delivered.",
                "pass": True,
                "score": 1.0,
            }
        ),
    )

    with capture_run_messages() as messages:
        asyncio.run(evaluator.evaluate(context))

    judge_prompt = _get_captured_judge_prompt(messages)

    assert context.inputs.user_message in judge_prompt
    assert context.output.agent_result.message in judge_prompt
    assert target_criterion.statement in judge_prompt


def test_response_criterion_evaluator_excludes_non_target_case_data_from_judge_request() -> (
    None
):
    target_criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    other_required_criterion = ResponseCriterion(
        id="states_delivery_date",
        statement="EXCLUDED other required response criterion.",
    )
    other_forbidden_criterion = ResponseCriterion(
        id="states_tracking_number",
        statement="EXCLUDED other forbidden response criterion.",
    )
    context = _create_response_evaluator_context(
        criterion=target_criterion,
        criterion_kind="required",
        response="Order-003 has been delivered.",
    )
    context.inputs = OrderEvalInput(
        user_message="What is the status of order-003?",
        customer_id="EXCLUDED-CUSTOMER-ID",
        execution_condition="shipment_information_failure",
    )
    context.metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="EXCLUDED_REQUIRED_TOOL",
                expected_arguments={
                    "order_id": "EXCLUDED-EXPECTED-ARGUMENT",
                },
                expected_outcome="shipment_not_found",
            ),
        ),
        forbidden_tools=frozenset({"EXCLUDED_FORBIDDEN_TOOL"}),
        required_tool_sequence=("EXCLUDED_REQUIRED_TOOL",),
        required_response_criteria=(
            target_criterion,
            other_required_criterion,
        ),
        forbidden_response_criteria=(other_forbidden_criterion,),
    )
    context.output = OrderEvalOutput(
        agent_result=AgentResult(
            message="Order-003 has been delivered.",
        ),
        tool_uses=(
            ObservedToolUse(
                tool_name="EXCLUDED_OBSERVED_TOOL",
                arguments={
                    "order_id": "EXCLUDED-OBSERVED-ARGUMENT",
                },
                outcome="tool_execution_failed",
            ),
        ),
    )

    recorded_span = SpanNode(
        name="EXCLUDED_TRACE_SPAN",
        trace_id=1,
        span_id=1,
        parent_span_id=None,
        start_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        end_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        attributes={
            "excluded.trace.attribute": "EXCLUDED-TRACE-VALUE",
        },
    )
    context.span_tree.add_spans([recorded_span])
    context.attributes = {
        "excluded.evaluation.attribute": "EXCLUDED-EVALUATION-VALUE",
    }

    evaluator = ResponseCriterionEvaluator(
        criterion=target_criterion,
        criterion_kind="required",
        judge_model=TestModel(
            custom_output_args={
                "reason": "The response states that the order was delivered.",
                "pass": True,
                "score": 1.0,
            }
        ),
    )

    with capture_run_messages() as messages:
        asyncio.run(evaluator.evaluate(context))

    judge_prompt = _get_captured_judge_prompt(messages)

    excluded_values = (
        other_required_criterion.id,
        other_required_criterion.statement,
        other_forbidden_criterion.id,
        other_forbidden_criterion.statement,
        context.inputs.customer_id,
        context.inputs.execution_condition,
        "EXCLUDED_REQUIRED_TOOL",
        "EXCLUDED_FORBIDDEN_TOOL",
        "EXCLUDED-EXPECTED-ARGUMENT",
        "shipment_not_found",
        "EXCLUDED_OBSERVED_TOOL",
        "EXCLUDED-OBSERVED-ARGUMENT",
        "tool_execution_failed",
        "EXCLUDED_TRACE_SPAN",
        "EXCLUDED-TRACE-VALUE",
        "EXCLUDED-EVALUATION-VALUE",
    )

    for excluded_value in excluded_values:
        assert excluded_value not in judge_prompt
