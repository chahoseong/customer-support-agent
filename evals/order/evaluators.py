from collections import Counter
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)

from evals.order.models import (
    ObservedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)


def _find_tool_uses(
    tool_uses: tuple[ObservedToolUse, ...],
    tool_name: str,
) -> tuple[ObservedToolUse, ...]:
    return tuple(tool_use for tool_use in tool_uses if tool_use.tool_name == tool_name)


def _filter_tool_names_for_trajectory(
    tool_uses: tuple[ObservedToolUse, ...],
    required_tool_sequence: tuple[str, ...],
) -> tuple[str, ...]:
    required_tool_names = set(required_tool_sequence)

    return tuple(
        tool_use.tool_name
        for tool_use in tool_uses
        if tool_use.tool_name in required_tool_names
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


@dataclass(repr=False)
class ToolArgumentsEvaluator(
    Evaluator[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ]
):
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

        issues: list[str] = []

        for expected_tool_use in metadata.required_tool_uses:
            matching_tool_uses = _find_tool_uses(
                ctx.output.tool_uses,
                expected_tool_use.tool_name,
            )

            if not matching_tool_uses:
                issues.append(
                    "Missing Tool observations for argument evaluation: "
                    f"{expected_tool_use.tool_name}."
                )
                continue

            for observed_tool_use in matching_tool_uses:
                actual_arguments = observed_tool_use.arguments

                if type(actual_arguments) is not dict:
                    issues.append(
                        f"Uninterpretable arguments for {expected_tool_use.tool_name}."
                    )
                    continue

                missing_argument_names = tuple(
                    argument_name
                    for argument_name in expected_tool_use.expected_arguments
                    if argument_name not in actual_arguments
                )

                mismatched_argument_names = tuple(
                    argument_name
                    for argument_name, expected_value in expected_tool_use.expected_arguments.items()
                    if argument_name in actual_arguments
                    and actual_arguments[argument_name] != expected_value
                )

                if missing_argument_names:
                    issues.append(
                        "Missing expected arguments for "
                        f"{expected_tool_use.tool_name}: "
                        f"{', '.join(missing_argument_names)}."
                    )

                if mismatched_argument_names:
                    issues.append(
                        "Mismatched expected arguments for "
                        f"{expected_tool_use.tool_name}: "
                        f"{', '.join(mismatched_argument_names)}."
                    )

        return {
            "tool_arguments": EvaluationReason(
                value=not issues,
                reason=" ".join(issues) or None,
            )
        }


@dataclass(repr=False)
class ToolOutcomesEvaluator(
    Evaluator[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ]
):
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

        issues: list[str] = []

        for expected_tool_use in metadata.required_tool_uses:
            matching_tool_uses = _find_tool_uses(
                ctx.output.tool_uses,
                expected_tool_use.tool_name,
            )

            if not matching_tool_uses:
                issues.append(
                    "Missing Tool observations for outcome evaluation: "
                    f"{expected_tool_use.tool_name}."
                )
                continue

            for observed_tool_use in matching_tool_uses:
                if observed_tool_use.outcome == expected_tool_use.expected_outcome:
                    continue

                issues.append(
                    f"Tool outcome mismatch for {expected_tool_use.tool_name}: "
                    f"expected {expected_tool_use.expected_outcome}; "
                    f"observed {observed_tool_use.outcome}."
                )

        return {
            "tool_outcomes": EvaluationReason(
                value=not issues,
                reason=" ".join(issues) or None,
            )
        }


@dataclass(repr=False)
class ToolTrajectoryEvaluator(
    Evaluator[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ]
):
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

        if not metadata.required_tool_sequence:
            return {}

        expected_tool_sequence = metadata.required_tool_sequence
        actual_tool_sequence = _filter_tool_names_for_trajectory(
            ctx.output.tool_uses,
            expected_tool_sequence,
        )

        matches_expected_order = actual_tool_sequence == expected_tool_sequence

        return {
            "tool_trajectory": EvaluationReason(
                value=matches_expected_order,
                reason=(
                    None
                    if matches_expected_order
                    else (
                        "Incorrect Tool order: "
                        f"expected {' -> '.join(expected_tool_sequence)}; "
                        f"observed {' -> '.join(actual_tool_sequence)}."
                    )
                ),
            )
        }
