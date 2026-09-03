from collections import Counter
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)

from evals.order.models import (
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)


@dataclass(repr=False)
class ToolSelectionEvaluator(
    Evaluator[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ]
):
    @staticmethod
    def _build_evaluation_reason(
        *,
        missing_tool_names: tuple[str, ...],
        repeated_tool_names: tuple[str, ...],
        forbidden_tool_names: tuple[str, ...],
        unexpected_tool_names: tuple[str, ...],
    ) -> EvaluationReason:
        issues: list[str] = []

        if missing_tool_names:
            issues.append(f"Missing required tools: {', '.join(missing_tool_names)}.")

        if repeated_tool_names:
            issues.append(f"Repeated required tools: {', '.join(repeated_tool_names)}.")

        if forbidden_tool_names:
            issues.append(f"Used forbidden tools: {', '.join(forbidden_tool_names)}.")

        if unexpected_tool_names:
            issues.append(f"Used unexpected tools: {', '.join(unexpected_tool_names)}.")

        return EvaluationReason(
            value=not issues,
            reason=" ".join(issues) or None,
        )

    def evaluate(
        self,
        ctx: EvaluatorContext[
            OrderEvalInput,
            OrderEvalOutput,
            OrderEvalMetadata,
        ],
    ) -> dict[str, EvaluationReason]:
        metadata = ctx.metadata

        if metadata is None:
            raise ValueError("Order evaluation metadata is required.")

        actual_tool_counts = Counter(
            tool_use.tool_name for tool_use in ctx.output.tool_uses
        )

        required_tool_names = {
            expected_tool_use.tool_name
            for expected_tool_use in metadata.required_tool_uses
        }

        missing_tool_names = tuple(
            expected_tool_use.tool_name
            for expected_tool_use in metadata.required_tool_uses
            if actual_tool_counts[expected_tool_use.tool_name] == 0
        )

        repeated_tool_names = tuple(
            expected_tool_use.tool_name
            for expected_tool_use in metadata.required_tool_uses
            if actual_tool_counts[expected_tool_use.tool_name] > 1
        )

        forbidden_tool_names = tuple(
            sorted(
                tool_name
                for tool_name in metadata.forbidden_tools
                if actual_tool_counts[tool_name] > 0
            )
        )

        unexpected_tool_names = tuple(
            sorted(
                tool_name
                for tool_name in actual_tool_counts
                if tool_name not in required_tool_names
                and tool_name not in metadata.forbidden_tools
            )
        )

        return {
            "tool_selection": self._build_evaluation_reason(
                missing_tool_names=missing_tool_names,
                repeated_tool_names=repeated_tool_names,
                forbidden_tool_names=forbidden_tool_names,
                unexpected_tool_names=unexpected_tool_names,
            ),
        }
