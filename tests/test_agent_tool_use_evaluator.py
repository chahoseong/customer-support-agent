from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from evals.order.evaluators import AgentToolUseEvaluator
from evals.order.models import (
    InformationSource,
    InformationSourceExpectation,
    InformationSourceOutcome,
    OrderEvalInput,
    OrderEvalMetadata,
)
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext
from pydantic_evals.otel import SpanNode, SpanTree

from customer_support_agent.agent import AgentResult
from customer_support_agent.tools.order import find_order, get_customer_orders
from customer_support_agent.tools.policy import get_cancellation_policy
from customer_support_agent.tools.shipment import find_shipment


@dataclass(frozen=True)
class _ToolSpanSpec:
    tool_name: str | None
    outcome: str | None = None
    error_code: str | None = None
    argument_names: str | list[str] | list[int] | None = None


def _create_tool_span_tree(*tool_span_specs: _ToolSpanSpec) -> SpanTree:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    tool_spans: list[SpanNode] = []

    for index, spec in enumerate(tool_span_specs, start=1):
        tool_span = SpanNode(
            name="tool.execute",
            trace_id=1,
            span_id=index,
            parent_span_id=None,
            start_timestamp=timestamp,
            end_timestamp=timestamp,
            attributes={},
        )

        if spec.tool_name is not None:
            tool_span.attributes["customer_support_agent.tool.name"] = spec.tool_name
        if spec.outcome is not None:
            tool_span.attributes["customer_support_agent.tool.outcome"] = spec.outcome
        if spec.error_code is not None:
            tool_span.attributes["customer_support_agent.tool.error.code"] = (
                spec.error_code
            )
        if spec.argument_names is not None:
            tool_span.attributes["customer_support_agent.tool.argument.names"] = (
                spec.argument_names
            )

        tool_spans.append(tool_span)

    span_tree = SpanTree()
    span_tree.add_spans(tool_spans)
    return span_tree


def _create_evaluator_context(
    *,
    metadata: OrderEvalMetadata,
    span_tree: SpanTree,
) -> EvaluatorContext[OrderEvalInput, AgentResult, OrderEvalMetadata]:
    return EvaluatorContext[
        OrderEvalInput,
        AgentResult,
        OrderEvalMetadata,
    ](
        name="agent-tool-use",
        inputs=OrderEvalInput(
            user_message="Help me with my order",
            customer_id="customer-001",
            execution_condition="default",
        ),
        metadata=metadata,
        expected_output=None,
        output=AgentResult(message="Here is the requested information."),
        duration=0.0,
        _span_tree=span_tree,
        attributes={},
        metrics={},
    )


def test_agent_tool_use_evaluator_returns_fixed_result_names() -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="customer_orders",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset({"shipment"}),
        required_response_criteria=("Report the customer's orders.",),
        forbidden_response_criteria=("Do not report shipment information",),
    )

    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=SpanTree(),
    )

    result = AgentToolUseEvaluator().evaluate(context)

    assert set(result) == {
        "agent_uses_required_information_sources",
        "agent_avoids_forbidden_information_sources",
        "agent_tool_calls_include_required_arguments",
        "agent_tool_calls_have_expected_outcomes",
    }


@pytest.mark.parametrize(
    ("tool_name", "source", "order_id"),
    [
        pytest.param(
            get_customer_orders.definition.name,
            "customer_orders",
            None,
            id="customer-orders",
        ),
        pytest.param(
            find_order.definition.name,
            "order",
            "order-001",
            id="order",
        ),
        pytest.param(
            find_shipment.definition.name,
            "shipment",
            "order-001",
            id="shipment",
        ),
        pytest.param(
            get_cancellation_policy.definition.name,
            "cancellation_policy",
            None,
            id="cancellation-policy",
        ),
    ],
)
def test_agent_tool_use_evaluator_recognizes_required_information_source_from_tool_span(
    tool_name: str,
    source: InformationSource,
    order_id: str | None,
) -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source=source, order_id=order_id, outcome="available"
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the requested information",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(_ToolSpanSpec(tool_name=tool_name)),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_uses_required_information_sources"].value is True


def test_agent_tool_use_evaluator_reports_missing_required_information_sources_without_tool_spans() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                ),
                InformationSourceExpectation(
                    source="shipment",
                    order_id="order-001",
                    outcome="available",
                ),
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the requested information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=SpanTree(),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_uses_required_information_sources"] == EvaluationReason(
        value=False,
        reason="Missing required information sources: order, shipment.",
    )


def test_agent_tool_use_evaluator_returns_true_when_forbidden_information_sources_are_not_observed() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="customer_orders",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset({"shipment"}),
        required_response_criteria=("Report the customer's orders.",),
        forbidden_response_criteria=("Do not report shipment information.",),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(tool_name=get_customer_orders.definition.name),
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_avoids_forbidden_information_sources"] == EvaluationReason(
        value=True
    )


def test_agent_tool_use_evaluator_reports_observed_forbidden_information_sources() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="customer_orders",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset({"order", "shipment"}),
        required_response_criteria=("Report the customer's orders.",),
        forbidden_response_criteria=("Do not report order or shipment details.",),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(tool_name=get_customer_orders.definition.name),
            _ToolSpanSpec(tool_name=find_shipment.definition.name),
            _ToolSpanSpec(tool_name=find_order.definition.name),
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_avoids_forbidden_information_sources"] == EvaluationReason(
        value=False,
        reason="Observed forbidden information sources: order, shipment.",
    )


@pytest.mark.parametrize(
    ("tool_span_spec", "source", "order_id", "expected_outcome"),
    [
        pytest.param(
            _ToolSpanSpec(
                tool_name=get_customer_orders.definition.name,
                outcome="success",
            ),
            "customer_orders",
            None,
            "available",
            id="success-is-available",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="tool_error",
                error_code="order_not_found",
            ),
            "order",
            "order-001",
            "unavailable",
            id="order-not-found-is-unavailable",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=find_shipment.definition.name,
                outcome="tool_error",
                error_code="shipment_not_found",
            ),
            "shipment",
            "order-001",
            "unavailable",
            id="shipment-not-found-is-unavailable",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="exception",
            ),
            "order",
            "order-001",
            "execution_failed",
            id="exception-is-execution-failed",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=get_cancellation_policy.definition.name,
                outcome="tool_error",
                error_code="tool_execution_failed",
            ),
            "cancellation_policy",
            None,
            "execution_failed",
            id="tool-execution-failed-code-is-execution-failed",
        ),
    ],
)
def test_agent_tool_use_evaluator_returns_true_when_tool_outcome_matches_expectation(
    tool_span_spec: _ToolSpanSpec,
    source: InformationSource,
    order_id: str | None,
    expected_outcome: InformationSourceOutcome,
) -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source=source,
                    order_id=order_id,
                    outcome=expected_outcome,
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the requested information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(tool_span_spec),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_have_expected_outcomes"] == EvaluationReason(
        value=True
    )


def test_agent_tool_use_evaluator_reports_tool_outcome_mismatch() -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="unavailable",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report that the order is unavailable.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
            )
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_have_expected_outcomes"] == EvaluationReason(
        value=False,
        reason=(
            "Tool outcome mismatch for order: expected unavailable; observed available."
        ),
    )


@pytest.mark.parametrize(
    ("tool_span_specs", "expected_result"),
    [
        pytest.param(
            (
                _ToolSpanSpec(
                    tool_name=find_order.definition.name,
                    outcome="success",
                ),
                _ToolSpanSpec(
                    tool_name=find_order.definition.name,
                    outcome="success",
                ),
            ),
            EvaluationReason(value=True),
            id="all-outcomes-match",
        ),
        pytest.param(
            (
                _ToolSpanSpec(
                    tool_name=find_order.definition.name,
                    outcome="success",
                ),
                _ToolSpanSpec(
                    tool_name=find_order.definition.name,
                    outcome="tool_error",
                    error_code="order_not_found",
                ),
            ),
            EvaluationReason(
                value=False,
                reason=(
                    "Tool outcome mismatch for order: "
                    "expected available; observed available, unavailable."
                ),
            ),
            id="one-outcome-mismatches",
        ),
    ],
)
def test_agent_tool_use_evaluator_requires_every_repeated_tool_outcome_to_match_expectation(
    tool_span_specs: tuple[_ToolSpanSpec, ...],
    expected_result: EvaluationReason,
) -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(*tool_span_specs),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_have_expected_outcomes"] == expected_result


@pytest.mark.parametrize(
    ("tool_span_spec", "expected_reason"),
    [
        pytest.param(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="tool_error",
                error_code="invalid_arguments",
            ),
            "Uninterpretable Tool observation for order: invalid_arguments.",
            id="invalid-arguments",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name="unrecognized_tool",
                outcome="tool_error",
                error_code="unknown_tool",
            ),
            (
                "Missing Tool outcome observations for information sources: order. "
                "Uninterpretable Tool observation: unrecognized Tool."
            ),
            id="unrecognized-tool",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="tool_error",
                error_code="shipment_not_found",
            ),
            (
                "Uninterpretable Tool observation for order: "
                "shipment_not_found does not match the information source."
            ),
            id="not-found-code-does-not-match-source",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
            ),
            "Uninterpretable Tool observation for order: missing outcome.",
            id="missing-outcome",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="unsupported",
            ),
            "Uninterpretable Tool observation for order: unsupported outcome.",
            id="unsupported-outcome",
        ),
        pytest.param(
            _ToolSpanSpec(
                tool_name=None,
                outcome="success",
            ),
            (
                "Missing Tool outcome observations for information sources: order. "
                "Uninterpretable Tool observation: missing Tool name."
            ),
            id="missing-tool-name",
        ),
    ],
)
def test_agent_tool_use_evaluator_reports_uninterpretable_tool_observation(
    tool_span_spec: _ToolSpanSpec,
    expected_reason: str,
) -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(tool_span_spec),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_have_expected_outcomes"] == EvaluationReason(
        value=False,
        reason=expected_reason,
    )


def test_agent_tool_use_evaluator_reports_missing_tool_outcome_observations() -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                ),
                InformationSourceExpectation(
                    source="shipment",
                    order_id="order-001",
                    outcome="available",
                ),
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order and shipment information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
            )
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_have_expected_outcomes"] == EvaluationReason(
        value=False,
        reason=("Missing Tool outcome observations for information sources: shipment."),
    )


@pytest.mark.parametrize(
    "argument_names",
    [
        pytest.param(["order_id"], id="required-name-only"),
        pytest.param(
            ["include_history", "order_id"],
            id="required-name-with-additional-name",
        ),
        pytest.param(
            '["include_history", "order_id"]',
            id="required-name-from-json-array",
        ),
    ],
)
def test_agent_tool_use_evaluator_returns_true_when_required_argument_names_are_present(
    argument_names: str | list[str],
) -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
                argument_names=argument_names,
            )
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_include_required_arguments"] == EvaluationReason(
        value=True
    )


def test_agent_tool_use_evaluator_reports_missing_required_argument_names_from_repeated_tool_calls() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
                argument_names=["order_id"],
            ),
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
                argument_names=["include_history"],
            ),
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_include_required_arguments"] == EvaluationReason(
        value=False,
        reason=(
            "Missing required Tool argument names for "
            "information sources: order (order_id)."
        ),
    )


@pytest.mark.parametrize(
    "argument_names",
    [
        pytest.param(None, id="missing-attribute"),
        pytest.param("order_id", id="invalid-json-string"),
        pytest.param([1, 2], id="non-string-list"),
        pytest.param('{"order_id": true}', id="json-non-array"),
    ],
)
def test_agent_tool_use_evaluator_reports_uninterpretable_argument_names(
    argument_names: str | list[int] | None,
) -> None:
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
                argument_names=argument_names,
            )
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_include_required_arguments"] == EvaluationReason(
        value=False,
        reason=("Uninterpretable Tool argument names for information sources: order."),
    )


def test_agent_tool_use_evaluator_returns_true_for_arguments_without_relevant_tool_calls() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=SpanTree(),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_include_required_arguments"] == EvaluationReason(
        value=True
    )


def test_agent_tool_use_evaluator_reports_missing_required_argument_names_in_source_order() -> (
    None
):
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="shipment",
                    order_id="order-001",
                    outcome="available",
                ),
                InformationSourceExpectation(
                    source="order",
                    order_id="order-001",
                    outcome="available",
                ),
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order and shipment information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(
                tool_name=find_shipment.definition.name,
                outcome="success",
                argument_names=[],
            ),
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
                argument_names=[],
            ),
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_include_required_arguments"] == EvaluationReason(
        value=False,
        reason=(
            "Missing required Tool argument names for information sources: "
            "order (order_id), shipment (order_id)."
        ),
    )


def test_agent_tool_use_evaluator_excludes_order_id_values_from_failure_reasons() -> (
    None
):
    sensitive_order_id = "sensitive-order-id-must-not-appear"
    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_information_sources=frozenset(
            {
                InformationSourceExpectation(
                    source="order",
                    order_id=sensitive_order_id,
                    outcome="available",
                )
            }
        ),
        forbidden_information_sources=frozenset(),
        required_response_criteria=("Report the order information.",),
        forbidden_response_criteria=(),
    )
    context = _create_evaluator_context(
        metadata=metadata,
        span_tree=_create_tool_span_tree(
            _ToolSpanSpec(
                tool_name=find_order.definition.name,
                outcome="success",
                argument_names=[],
            )
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_tool_calls_include_required_arguments"].value is False
    assert all(
        result.reason is None or sensitive_order_id not in result.reason
        for result in results.values()
    )
