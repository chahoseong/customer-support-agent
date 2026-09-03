import pytest
from evals.order.evaluators import (
    ToolArgumentsEvaluator,
    ToolOutcomesEvaluator,
    ToolSelectionEvaluator,
    ToolTrajectoryEvaluator,
)
from evals.order.models import (
    ExpectedToolUse,
    ObservedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel import SpanTree

from customer_support_agent.agent import AgentResult


def _create_evaluator_context(
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
        name="tool-evaluator-independence",
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
            required_tool_sequence=("find_order", "find_shipment"),
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


def _evaluate_tool_use_axes(
    context: EvaluatorContext[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ],
) -> dict[str, bool]:
    evaluation_results = {
        **ToolSelectionEvaluator().evaluate(context),
        **ToolArgumentsEvaluator().evaluate(context),
        **ToolOutcomesEvaluator().evaluate(context),
        **ToolTrajectoryEvaluator().evaluate(context),
    }

    evaluation_values: dict[str, bool] = {}

    for name, result in evaluation_results.items():
        value = result.value
        assert isinstance(value, bool)
        evaluation_values[name] = value

    return evaluation_values


@pytest.mark.parametrize(
    ("tool_uses", "expected_results"),
    [
        pytest.param(
            (
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
            {
                "tool_selection": True,
                "tool_arguments": True,
                "tool_outcomes": True,
                "tool_trajectory": True,
            },
            id="all-axes-pass",
        ),
        pytest.param(
            (
                ObservedToolUse(
                    tool_name="find_order",
                    arguments={"order_id": "order-002"},
                    outcome="success",
                ),
                ObservedToolUse(
                    tool_name="search_web",
                    arguments={"query": "shipment tracking"},
                    outcome="success",
                ),
                ObservedToolUse(
                    tool_name="find_shipment",
                    arguments={"order_id": "order-002"},
                    outcome="success",
                ),
            ),
            {
                "tool_selection": False,
                "tool_arguments": True,
                "tool_outcomes": True,
                "tool_trajectory": True,
            },
            id="selection-only-fails",
        ),
        pytest.param(
            (
                ObservedToolUse(
                    tool_name="find_order",
                    arguments={"order_id": "different-order"},
                    outcome="success",
                ),
                ObservedToolUse(
                    tool_name="find_shipment",
                    arguments={"order_id": "order-002"},
                    outcome="success",
                ),
            ),
            {
                "tool_selection": True,
                "tool_arguments": False,
                "tool_outcomes": True,
                "tool_trajectory": True,
            },
            id="arguments-only-fails",
        ),
        pytest.param(
            (
                ObservedToolUse(
                    tool_name="find_order",
                    arguments={"order_id": "order-002"},
                    outcome="order_not_found",
                ),
                ObservedToolUse(
                    tool_name="find_shipment",
                    arguments={"order_id": "order-002"},
                    outcome="success",
                ),
            ),
            {
                "tool_selection": True,
                "tool_arguments": True,
                "tool_outcomes": False,
                "tool_trajectory": True,
            },
            id="outcomes-only-fails",
        ),
        pytest.param(
            (
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
            {
                "tool_selection": True,
                "tool_arguments": True,
                "tool_outcomes": True,
                "tool_trajectory": False,
            },
            id="trajectory-only-fails",
        ),
    ],
)
def test_tool_evaluators_keep_axis_results_independent(
    tool_uses: tuple[ObservedToolUse, ...],
    expected_results: dict[str, bool],
) -> None:
    context = _create_evaluator_context(tool_uses)

    results = _evaluate_tool_use_axes(context)

    assert results == expected_results
