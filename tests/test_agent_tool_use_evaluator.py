from datetime import UTC, datetime

import pytest
from evals.order.evaluators import AgentToolUseEvaluator
from evals.order.models import (
    InformationSource,
    InformationSourceExpectation,
    OrderEvalInput,
    OrderEvalMetadata,
)
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext
from pydantic_evals.otel import SpanNode, SpanTree

from customer_support_agent.agent import AgentResult
from customer_support_agent.tools.order import find_order, get_customer_orders
from customer_support_agent.tools.policy import get_cancellation_policy
from customer_support_agent.tools.shipment import find_shipment


def _create_tool_span_tree(*tool_names: str) -> SpanTree:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    tool_spans = [
        SpanNode(
            name="tool.execute",
            trace_id=1,
            span_id=index,
            parent_span_id=None,
            start_timestamp=timestamp,
            end_timestamp=timestamp,
            attributes={
                "customer_support_agent.tool.name": tool_name,
            },
        )
        for index, tool_name in enumerate(tool_names, start=1)
    ]
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
        metadata=metadata, span_tree=_create_tool_span_tree(tool_name)
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
            get_customer_orders.definition.name,
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
            get_customer_orders.definition.name,
            find_shipment.definition.name,
            find_order.definition.name,
        ),
    )

    results = AgentToolUseEvaluator().evaluate(context)

    assert results["agent_avoids_forbidden_information_sources"] == EvaluationReason(
        value=False,
        reason="Observed forbidden information sources: order, shipment.",
    )
