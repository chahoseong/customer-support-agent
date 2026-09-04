from dataclasses import dataclass
from typing import Literal

from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output

from evals.order.models import (
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
    ResponseCriterion,
)


@dataclass(repr=False)
class ResponseCriterionEvaluator(
    Evaluator[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ]
):
    criterion: ResponseCriterion
    criterion_kind: Literal["required"]
    judge_model: Model | KnownModelName | str
    judge_model_settings: ModelSettings | None = None

    def __post_init__(self) -> None:
        if not self.judge_model:
            raise ValueError("Judge model is required.")

    def get_default_evaluation_name(self) -> str:
        return f"response_{self.criterion_kind}_{self.criterion.id}"

    async def evaluate(
        self,
        ctx: EvaluatorContext[
            OrderEvalInput,
            OrderEvalOutput,
            OrderEvalMetadata,
        ],
    ) -> EvaluationReason:
        grading = await judge_input_output(
            inputs=ctx.inputs.user_message,
            output=ctx.output.agent_result.message,
            rubric=self.criterion.statement,
            model=self.judge_model,
            model_settings=self.judge_model_settings,
        )

        return EvaluationReason(
            value=grading.pass_,
            reason=grading.reason,
        )
