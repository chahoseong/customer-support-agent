from pathlib import Path

from pydantic_evals import Dataset

from .evaluators import (
    ToolArgumentsEvaluator,
    ToolOutcomesEvaluator,
    ToolSelectionEvaluator,
    ToolTrajectoryEvaluator,
)
from .models import (
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
)

_DATASET_PATH = Path(__file__).with_name("scenario_cases.yaml")


def load_order_dataset() -> Dataset[OrderEvalInput, OrderEvalOutput, OrderEvalMetadata]:
    dataset = Dataset[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ].from_file(_DATASET_PATH)

    dataset.add_evaluator(ToolSelectionEvaluator())
    dataset.add_evaluator(ToolArgumentsEvaluator())
    dataset.add_evaluator(ToolOutcomesEvaluator())
    dataset.add_evaluator(ToolTrajectoryEvaluator())

    return dataset
