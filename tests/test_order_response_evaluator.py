import asyncio
from typing import cast

import pytest
from evals.order.models import (
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
    ResponseCriterion,
)
from evals.order.response_evaluator import ResponseCriterionEvaluator
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_evals.evaluators import (
    EvaluationReason,
    EvaluatorContext,
)
from pydantic_evals.otel import SpanTree

from customer_support_agent.agent import AgentResult


def _create_response_evaluator_context(
    *,
    criterion: ResponseCriterion,
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
        name="required-response-criterion",
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
            required_response_criteria=(criterion,),
            forbidden_response_criteria=(),
        ),
        expected_output=None,
        output=OrderEvalOutput(
            agent_result=AgentResult(
                message=response,
            ),
            tool_uses=(),
        ),
        duration=0.0,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
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
