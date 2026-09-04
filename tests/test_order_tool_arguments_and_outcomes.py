import pytest
from evals.order.models import (
    ExpectedToolOutcome,
    ExpectedToolUse,
    ObservedToolOutcome,
    ObservedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)
from evals.order.tool_evaluators import (
    ToolArgumentsEvaluator,
    ToolOutcomesEvaluator,
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
        name="tool-arguments-and-outcomes",
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


def test_tool_arguments_evaluator_returns_true_when_expected_arguments_are_subset() -> (
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
                arguments={
                    "order_id": "order-002",
                    "include_history": True,
                },
                outcome="success",
            ),
        ),
    )

    result = ToolArgumentsEvaluator().evaluate(context)

    assert result == {
        "tool_arguments": EvaluationReason(value=True),
    }


def test_tool_arguments_evaluator_reports_missing_expected_argument() -> None:
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
                arguments={"include_history": True},
                outcome="success",
            ),
        ),
    )

    result = ToolArgumentsEvaluator().evaluate(context)
    arguments = result["tool_arguments"]

    assert arguments.value is False
    assert arguments.reason is not None
    assert "missing" in arguments.reason.lower()
    assert "find_order" in arguments.reason
    assert "order_id" in arguments.reason


def test_tool_arguments_evaluator_reports_mismatched_argument_value() -> None:
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
                outcome="success",
            ),
        ),
    )

    result = ToolArgumentsEvaluator().evaluate(context)
    arguments = result["tool_arguments"]

    assert arguments.value is False
    assert arguments.reason is not None
    assert "mismatch" in arguments.reason.lower()
    assert "find_order" in arguments.reason
    assert "order_id" in arguments.reason


def test_tool_arguments_evaluator_omits_expected_and_observed_values_from_failure_reason() -> (
    None
):
    expected_argument_value = "sensitive-expected-order-id"
    observed_argument_value = "sensitive-observed-order-id"
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="find_order",
                expected_arguments={"order_id": expected_argument_value},
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
                arguments={"order_id": observed_argument_value},
                outcome="success",
            ),
        ),
    )

    result = ToolArgumentsEvaluator().evaluate(context)
    arguments = result["tool_arguments"]

    assert arguments.value is False
    assert arguments.reason is not None
    assert expected_argument_value not in arguments.reason
    assert observed_argument_value not in arguments.reason


def test_tool_arguments_evaluator_reports_uninterpretable_arguments_when_actual_is_not_dict() -> (
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
                arguments="order-002",
                outcome="success",
            ),
        ),
    )

    result = ToolArgumentsEvaluator().evaluate(context)
    arguments = result["tool_arguments"]

    assert arguments.value is False
    assert arguments.reason is not None
    assert "uninterpretable" in arguments.reason.lower()
    assert "find_order" in arguments.reason


def test_tool_arguments_evaluator_reports_missing_required_tool_observation() -> None:
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
        tool_uses=(),
    )

    result = ToolArgumentsEvaluator().evaluate(context)
    arguments = result["tool_arguments"]

    assert arguments.value is False
    assert arguments.reason is not None
    assert "missing" in arguments.reason.lower()
    assert "find_order" in arguments.reason


@pytest.mark.parametrize(
    ("second_arguments", "expected_value"),
    [
        pytest.param(
            {"order_id": "order-002"},
            True,
            id="all-observations-match",
        ),
        pytest.param(
            {"order_id": "different-order"},
            False,
            id="one-observation-mismatches",
        ),
    ],
)
def test_tool_arguments_evaluator_requires_every_repeated_tool_observation_to_match(
    second_arguments: dict[str, object],
    expected_value: bool,
) -> None:
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
                arguments=second_arguments,
                outcome="success",
            ),
        ),
    )

    result = ToolArgumentsEvaluator().evaluate(context)

    assert result["tool_arguments"].value is expected_value


@pytest.mark.parametrize(
    "expected_outcome",
    [
        pytest.param("success", id="success"),
        pytest.param("invalid_arguments", id="invalid-arguments"),
        pytest.param("order_not_found", id="order-not-found"),
        pytest.param("shipment_not_found", id="shipment-not-found"),
        pytest.param("tool_execution_failed", id="tool-execution-failed"),
        pytest.param("unknown_tool", id="unknown-tool"),
    ],
)
def test_tool_outcomes_evaluator_returns_true_when_outcome_matches(
    expected_outcome: ExpectedToolOutcome,
) -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(
            ExpectedToolUse(
                tool_name="find_order",
                expected_arguments={"order_id": "order-002"},
                expected_outcome=expected_outcome,
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
                outcome=expected_outcome,
            ),
        ),
    )

    result = ToolOutcomesEvaluator().evaluate(context)

    assert result == {
        "tool_outcomes": EvaluationReason(value=True),
    }


def test_tool_outcomes_evaluator_reports_mismatched_outcome() -> None:
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
                outcome="order_not_found",
            ),
        ),
    )

    result = ToolOutcomesEvaluator().evaluate(context)
    outcomes = result["tool_outcomes"]

    assert outcomes.value is False
    assert outcomes.reason is not None
    assert "mismatch" in outcomes.reason.lower()
    assert "find_order" in outcomes.reason
    assert "success" in outcomes.reason
    assert "order_not_found" in outcomes.reason


def test_tool_outcomes_evaluator_reports_missing_required_tool_observation() -> None:
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
        tool_uses=(),
    )

    result = ToolOutcomesEvaluator().evaluate(context)
    outcomes = result["tool_outcomes"]

    assert outcomes.value is False
    assert outcomes.reason is not None
    assert "missing" in outcomes.reason.lower()
    assert "find_order" in outcomes.reason


@pytest.mark.parametrize(
    ("second_outcome", "expected_value"),
    [
        pytest.param(
            "success",
            True,
            id="all-observations-match",
        ),
        pytest.param(
            "order_not_found",
            False,
            id="one-observation-mismatches",
        ),
    ],
)
def test_tool_outcomes_evaluator_requires_every_repeated_tool_observation_to_match(
    second_outcome: ObservedToolOutcome,
    expected_value: bool,
) -> None:
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
                outcome=second_outcome,
            ),
        ),
    )

    result = ToolOutcomesEvaluator().evaluate(context)

    assert result["tool_outcomes"].value is expected_value
