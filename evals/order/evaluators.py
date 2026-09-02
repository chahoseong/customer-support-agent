from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)

from customer_support_agent.agent import AgentResult

from .models import (
    InformationSource,
    OrderEvalInput,
    OrderEvalMetadata,
)

_TOOL_NAME_ATTRIBUTE = "customer_support_agent.tool.name"

_INFORMATION_SOURCE_BY_TOOL_NAME: dict[str, InformationSource] = {
    "get_customer_orders": "customer_orders",
    "find_order": "order",
    "find_shipment": "shipment",
    "get_cancellation_policy": "cancellation_policy",
}


@dataclass
class AgentToolUseEvaluator(Evaluator[OrderEvalInput, AgentResult, OrderEvalMetadata]):
    def evaluate(
        self,
        ctx: EvaluatorContext[
            OrderEvalInput,
            AgentResult,
            OrderEvalMetadata,
        ],
    ) -> Mapping[str, EvaluationReason]:
        metadata = ctx.metadata
        if metadata is None:
            raise ValueError("Order evaluation metadata is required.")

        observed_sources: set[InformationSource] = set()

        for span in ctx.span_tree.find({"name_equals": "tool.execute"}):
            tool_name = span.attributes.get(_TOOL_NAME_ATTRIBUTE)
            if not isinstance(tool_name, str):
                continue

            information_source = _INFORMATION_SOURCE_BY_TOOL_NAME.get(tool_name)
            if information_source is not None:
                observed_sources.add(information_source)

        required_sources = {
            expectation.source for expectation in metadata.required_information_sources
        }
        missing_required_sources = required_sources - observed_sources
        uses_required_sources = not missing_required_sources
        missing_required_sources_reason = (
            None
            if uses_required_sources
            else (
                "Missing required information sources: "
                f"{', '.join(sorted(missing_required_sources))}."
            )
        )

        observed_forbidden_sources = (
            metadata.forbidden_information_sources & observed_sources
        )
        avoids_forbidden_sources = not observed_forbidden_sources
        observed_forbidden_sources_reason = (
            None
            if avoids_forbidden_sources
            else (
                "Observed forbidden information sources: "
                f"{', '.join(sorted(observed_forbidden_sources))}."
            )
        )

        return {
            "agent_uses_required_information_sources": EvaluationReason(
                value=uses_required_sources,
                reason=missing_required_sources_reason,
            ),
            "agent_avoids_forbidden_information_sources": EvaluationReason(
                value=avoids_forbidden_sources,
                reason=observed_forbidden_sources_reason,
            ),
            "agent_tool_calls_include_required_arguments": EvaluationReason(
                value=False,
                reason="Evaluation is not implemented.",
            ),
            "agent_tool_calls_have_expected_outcomes": EvaluationReason(
                value=False,
                reason="Evaluation is not implemented.",
            ),
        }
