from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)
from pydantic_evals.otel import SpanNode

from customer_support_agent.agent import AgentResult

from .models import (
    InformationSource,
    InformationSourceExpectation,
    InformationSourceOutcome,
    OrderEvalInput,
    OrderEvalMetadata,
)

_TOOL_NAME_ATTRIBUTE = "customer_support_agent.tool.name"
_TOOL_OUTCOME_ATTRIBUTE = "customer_support_agent.tool.outcome"
_TOOL_ERROR_CODE_ATTRIBUTE = "customer_support_agent.tool.error.code"

_INFORMATION_SOURCE_BY_TOOL_NAME: dict[str, InformationSource] = {
    "get_customer_orders": "customer_orders",
    "find_order": "order",
    "find_shipment": "shipment",
    "get_cancellation_policy": "cancellation_policy",
}


@dataclass(frozen=True)
class _ToolObservation:
    tool_name: str | None
    information_source: InformationSource | None
    outcome: InformationSourceOutcome | None
    diagnostic: str | None


def _normalize_tool_outcome(
    *,
    information_source: InformationSource | None,
    tool_outcome: object,
    error_code: object,
) -> InformationSourceOutcome | None:
    if tool_outcome == "exception" or error_code == "tool_execution_failed":
        return "execution_failed"

    if tool_outcome == "success":
        return "available"

    if (
        tool_outcome == "tool_error"
        and information_source == "order"
        and error_code == "order_not_found"
    ):
        return "unavailable"

    if (
        tool_outcome == "tool_error"
        and information_source == "shipment"
        and error_code == "shipment_not_found"
    ):
        return "unavailable"

    return None


def _get_tool_observation_diagnostic(
    *,
    tool_name: str | None,
    information_source: InformationSource | None,
    tool_outcome: object,
    error_code: object,
    normalized_outcome: InformationSourceOutcome | None,
) -> str | None:
    if tool_name is None:
        return "Uninterpretable Tool observation: missing Tool name."

    if information_source is None:
        return "Uninterpretable Tool observation: unrecognized Tool."

    if normalized_outcome is not None:
        return None

    diagnostic_prefix = f"Uninterpretable Tool observation for {information_source}: "

    if error_code == "invalid_arguments":
        return f"{diagnostic_prefix}invalid_arguments."

    if error_code in {"order_not_found", "shipment_not_found"}:
        return f"{diagnostic_prefix}{error_code} does not match the information source."

    if tool_outcome is None:
        return f"{diagnostic_prefix}missing outcome."

    if tool_outcome == "tool_error" and error_code is None:
        return f"{diagnostic_prefix}missing error code."

    if tool_outcome == "tool_error":
        return f"{diagnostic_prefix}unsupported error code."

    return f"{diagnostic_prefix}unsupported outcome."


def _observe_tool_span(span: SpanNode) -> _ToolObservation:
    raw_tool_name = span.attributes.get(_TOOL_NAME_ATTRIBUTE)
    tool_name = raw_tool_name if isinstance(raw_tool_name, str) else None

    information_source = (
        _INFORMATION_SOURCE_BY_TOOL_NAME.get(tool_name)
        if tool_name is not None
        else None
    )
    tool_outcome = span.attributes.get(_TOOL_OUTCOME_ATTRIBUTE)
    error_code = span.attributes.get(_TOOL_ERROR_CODE_ATTRIBUTE)

    normalized_outcome = _normalize_tool_outcome(
        information_source=information_source,
        tool_outcome=tool_outcome,
        error_code=error_code,
    )

    return _ToolObservation(
        tool_name=tool_name,
        information_source=information_source,
        outcome=normalized_outcome,
        diagnostic=_get_tool_observation_diagnostic(
            tool_name=tool_name,
            information_source=information_source,
            tool_outcome=tool_outcome,
            error_code=error_code,
            normalized_outcome=normalized_outcome,
        ),
    )


def _evaluate_tool_outcomes(
    *,
    expectations: frozenset[InformationSourceExpectation],
    observations: list[_ToolObservation],
) -> EvaluationReason:
    observation_diagnostics = sorted(
        {
            observation.diagnostic
            for observation in observations
            if observation.diagnostic is not None
        }
    )

    missing_sources: set[InformationSource] = set()
    mismatch_reasons: list[str] = []

    for expectation in sorted(
        expectations,
        key=lambda expectation: expectation.source,
    ):
        source_observations = [
            observation
            for observation in observations
            if observation.information_source == expectation.source
        ]

        if not source_observations:
            missing_sources.add(expectation.source)
            continue

        if any(
            observation.outcome != expectation.outcome
            for observation in source_observations
        ) and all(
            observation.outcome is not None for observation in source_observations
        ):
            observed_outcomes = sorted(
                {
                    observation.outcome
                    for observation in source_observations
                    if observation.outcome is not None
                }
            )
            mismatch_reasons.append(
                f"Tool outcome mismatch for {expectation.source}: "
                f"expected {expectation.outcome}; "
                f"observed {', '.join(observed_outcomes)}."
            )

    failure_reasons: list[str] = []

    if missing_sources:
        failure_reasons.append(
            "Missing Tool outcome observations for information sources: "
            f"{', '.join(sorted(missing_sources))}."
        )

    failure_reasons.extend(observation_diagnostics)
    failure_reasons.extend(mismatch_reasons)

    return EvaluationReason(
        value=not failure_reasons,
        reason=" ".join(failure_reasons) if failure_reasons else None,
    )


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

        tool_observations = [
            _observe_tool_span(span)
            for span in ctx.span_tree.find({"name_equals": "tool.execute"})
        ]
        observed_sources = {
            observation.information_source
            for observation in tool_observations
            if observation.information_source is not None
        }

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

        tool_outcome_evaluation = _evaluate_tool_outcomes(
            expectations=metadata.required_information_sources,
            observations=tool_observations,
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
            "agent_tool_calls_have_expected_outcomes": tool_outcome_evaluation,
        }
