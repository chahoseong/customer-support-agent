from evals.order.models import (
    ExpectedToolUse,
    ObservedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)
from evals.order.tool_evaluators import ToolTrajectoryEvaluator
from pydantic_evals.evaluators import (
    EvaluationReason,
    EvaluatorContext,
)
from pydantic_evals.otel import SpanTree

from customer_support_agent.agent import AgentResult


def _create_evaluator_context(
    *,
    required_tool_sequence: tuple[str, ...],
    tool_uses: tuple[ObservedToolUse, ...],
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
        name="tool-trajectory",
        inputs=OrderEvalInput(
            user_message="Where is my order?",
            customer_id="customer-001",
            execution_condition="default",
        ),
        metadata=OrderEvalMetadata(
            scenario_id="scenario-1",
            required_tool_uses=(
                ExpectedToolUse(
                    tool_name="find_order",
                    expected_arguments={"order_id": "order-002"},
                    expected_outcome="success",
                ),
                ExpectedToolUse(
                    tool_name="find_shipment",
                    expected_arguments={"order_id": "order-002"},
                    expected_outcome="success",
                ),
            ),
            forbidden_tools=frozenset(),
            required_tool_sequence=required_tool_sequence,
            required_response_criteria=(),
            forbidden_response_criteria=(),
        ),
        expected_output=None,
        output=OrderEvalOutput(
            agent_result=AgentResult(message="Your order is in transit."),
            tool_uses=tool_uses,
        ),
        duration=0.0,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )


def test_tool_trajectory_evaluator_returns_true_when_required_tools_follow_relative_order() -> (
    None
):
    context = _create_evaluator_context(
        required_tool_sequence=("find_order", "find_shipment"),
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
            ObservedToolUse(
                tool_name="find_shipment",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
        ),
    )

    result = ToolTrajectoryEvaluator().evaluate(context)

    assert result == {
        "tool_trajectory": EvaluationReason(value=True),
    }


def test_tool_trajectory_evaluator_reports_incorrect_order_when_required_tools_are_reversed() -> (
    None
):
    context = _create_evaluator_context(
        required_tool_sequence=("find_order", "find_shipment"),
        tool_uses=(
            ObservedToolUse(
                tool_name="find_shipment",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
        ),
    )

    result = ToolTrajectoryEvaluator().evaluate(context)
    trajectory = result["tool_trajectory"]

    assert trajectory.value is False
    assert trajectory.reason is not None
    assert "order" in trajectory.reason.lower()
    assert "find_order" in trajectory.reason
    assert "find_shipment" in trajectory.reason


def test_tool_trajectory_evaluator_ignores_tool_not_in_required_sequence_when_checking_relative_order() -> (
    None
):
    context = _create_evaluator_context(
        required_tool_sequence=("find_order", "find_shipment"),
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
            ObservedToolUse(
                tool_name="search_knowledge_base",
                arguments={"query": "shipment tracking"},
                outcome="success",
            ),
            ObservedToolUse(
                tool_name="find_shipment",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
        ),
    )

    result = ToolTrajectoryEvaluator().evaluate(context)

    assert result == {
        "tool_trajectory": EvaluationReason(value=True),
    }


def test_tool_trajectory_evaluator_returns_no_result_when_relative_order_is_not_required() -> (
    None
):
    context = _create_evaluator_context(
        required_tool_sequence=(),
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
            ObservedToolUse(
                tool_name="find_shipment",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
        ),
    )

    result = ToolTrajectoryEvaluator().evaluate(context)

    assert result == {}
