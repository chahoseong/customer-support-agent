import pytest
from evals.order.evaluators import ToolSelectionEvaluator
from evals.order.models import (
    ExpectedToolUse,
    ObservedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)
from pydantic_evals.evaluators import (
    EvaluationReason,
    EvaluatorContext,
)
from pydantic_evals.otel import SpanTree

from customer_support_agent.agent import AgentResult


def _create_evaluator_context(
    *,
    metadata: OrderEvalMetadata | None,
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
        name="tool-selection",
        inputs=OrderEvalInput(
            user_message="Look up the order.",
            customer_id="customer-001",
            execution_condition="default",
        ),
        metadata=metadata,
        expected_output=None,
        output=OrderEvalOutput(
            agent_result=AgentResult(
                message="Here is the requested order.",
            ),
            tool_uses=tool_uses,
        ),
        duration=0.0,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )


def test_tool_selection_evaluator_returns_true_when_required_tools_are_used_once() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="find_order",
                expected_arguments={"order_id": "order-002"},
                expected_outcome="success",
            ),
        ),
        forbidden_tools=frozenset({"find_shipment"}),
        required_tool_sequence=(),
        required_response_criteria=(),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
        ),
    )

    result = ToolSelectionEvaluator().evaluate(context)

    assert result == {
        "tool_selection": EvaluationReason(value=True),
    }


def test_tool_selection_evaluator_reports_missing_required_tool() -> None:
    metadata = OrderEvalMetadata(
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
        required_tool_sequence=(),
        required_response_criteria=(),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
        ),
    )

    result = ToolSelectionEvaluator().evaluate(context)
    selection = result["tool_selection"]

    assert selection.value is False
    assert selection.reason is not None
    assert "missing" in selection.reason.lower()
    assert "find_shipment" in selection.reason


def test_tool_selection_evaluator_raises_when_metadata_is_missing() -> None:
    context = _create_evaluator_context(
        metadata=None,
        tool_uses=(),
    )

    with pytest.raises(
        ValueError,
        match="Order evaluation metadata is required",
    ):
        ToolSelectionEvaluator().evaluate(context)


def test_tool_selection_evaluator_reports_repeated_required_tool() -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="find_order",
                expected_arguments={"order_id": "order-002"},
                expected_outcome="success",
            ),
        ),
        forbidden_tools=frozenset(),
        required_tool_sequence=(),
        required_response_criteria=(),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
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

    result = ToolSelectionEvaluator().evaluate(context)
    selection = result["tool_selection"]

    assert selection.value is False
    assert selection.reason is not None
    assert "repeated" in selection.reason.lower()
    assert "find_order" in selection.reason


def test_tool_selection_evaluator_reports_forbidden_tool() -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="find_order",
                expected_arguments={"order_id": "order-002"},
                expected_outcome="success",
            ),
        ),
        forbidden_tools=frozenset({"find_shipment"}),
        required_tool_sequence=(),
        required_response_criteria=(),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
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

    result = ToolSelectionEvaluator().evaluate(context)
    selection = result["tool_selection"]

    assert selection.value is False
    assert selection.reason is not None
    assert "forbidden" in selection.reason.lower()
    assert "find_shipment" in selection.reason


def test_tool_selection_evaluator_reports_unexpected_tool() -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="find_order",
                expected_arguments={"order_id": "order-002"},
                expected_outcome="success",
            ),
        ),
        forbidden_tools=frozenset({"find_shipment"}),
        required_tool_sequence=(),
        required_response_criteria=(),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "order-002"},
                outcome="success",
            ),
            ObservedToolUse(
                tool_name="search_web",
                arguments={"query": "order-002"},
                outcome="success",
            ),
        ),
    )

    result = ToolSelectionEvaluator().evaluate(context)
    selection = result["tool_selection"]

    assert selection.value is False
    assert selection.reason is not None
    assert "unexpected" in selection.reason.lower()
    assert "search_web" in selection.reason


def test_tool_selection_evaluator_returns_true_when_only_arguments_and_outcome_differ() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="find_order",
                expected_arguments={"order_id": "order-002"},
                expected_outcome="success",
            ),
        ),
        forbidden_tools=frozenset(),
        required_tool_sequence=(),
        required_response_criteria=(),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        tool_uses=(
            ObservedToolUse(
                tool_name="find_order",
                arguments={"order_id": "different-order"},
                outcome="order_not_found",
            ),
        ),
    )

    result = ToolSelectionEvaluator().evaluate(context)

    assert result == {
        "tool_selection": EvaluationReason(value=True),
    }
