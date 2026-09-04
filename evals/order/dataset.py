from pathlib import Path

from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Dataset

from .models import (
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)
from .response_evaluator import ResponseCriterionEvaluator
from .tool_evaluators import (
    ToolArgumentsEvaluator,
    ToolOutcomesEvaluator,
    ToolSelectionEvaluator,
    ToolTrajectoryEvaluator,
)

_DATASET_PATH = Path(__file__).with_name("scenario_cases.yaml")


def _create_case_response_evaluators(
    *,
    metadata: OrderEvalMetadata,
    judge_model: Model | KnownModelName | str,
    judge_model_settings: ModelSettings | None,
) -> tuple[ResponseCriterionEvaluator, ...]:
    return (
        *(
            ResponseCriterionEvaluator(
                criterion=criterion,
                criterion_kind="required",
                judge_model=judge_model,
                judge_model_settings=judge_model_settings,
            )
            for criterion in metadata.required_response_criteria
        ),
        *(
            ResponseCriterionEvaluator(
                criterion=criterion,
                criterion_kind="forbidden",
                judge_model=judge_model,
                judge_model_settings=judge_model_settings,
            )
            for criterion in metadata.forbidden_response_criteria
        ),
    )


def load_order_dataset(
    *,
    judge_model: Model | KnownModelName | str,
    judge_model_settings: ModelSettings | None = None,
) -> Dataset[OrderEvalInput, OrderEvalOutput, OrderEvalMetadata]:
    dataset = Dataset[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ].from_file(_DATASET_PATH)

    dataset.add_evaluator(ToolSelectionEvaluator())
    dataset.add_evaluator(ToolArgumentsEvaluator())
    dataset.add_evaluator(ToolOutcomesEvaluator())
    dataset.add_evaluator(ToolTrajectoryEvaluator())

    for case in dataset.cases:
        if case.metadata is None:
            continue

        case.evaluators.extend(
            _create_case_response_evaluators(
                metadata=case.metadata,
                judge_model=judge_model,
                judge_model_settings=judge_model_settings,
            )
        )

    return dataset
