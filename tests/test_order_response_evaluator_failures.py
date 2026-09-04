import asyncio

from evals.order.models import (
    ExpectedToolUse,
    ObservedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
    OrderEvalOutput,
    ResponseCriterion,
)
from evals.order.response_evaluator import ResponseCriterionEvaluator
from evals.order.tool_evaluators import ToolSelectionEvaluator
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset

from customer_support_agent.agent import AgentResult


def test_response_criterion_evaluator_reports_judge_exception_as_evaluator_failure() -> (
    None
):
    def raise_judge_error(
        _messages: list[ModelMessage],
        _agent_info: AgentInfo,
    ) -> ModelResponse:
        raise RuntimeError("judge unavailable")

    criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    evaluator = ResponseCriterionEvaluator(
        criterion=criterion,
        criterion_kind="required",
        judge_model=FunctionModel(raise_judge_error),
    )
    dataset = Dataset[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ](
        name="judge_failure",
        cases=[
            Case(
                name="judge_failure",
                inputs=OrderEvalInput(
                    user_message="order-003은 지금 어떤 상태인가요?",
                    customer_id="customer-001",
                    execution_condition="default",
                ),
                evaluators=(evaluator,),
            )
        ],
    )

    def return_agent_output(_inputs: OrderEvalInput) -> OrderEvalOutput:
        return OrderEvalOutput(
            agent_result=AgentResult(message="Order-003 has been delivered."),
            tool_uses=(),
        )

    report = asyncio.run(
        dataset.evaluate(
            return_agent_output,
            progress=False,
        )
    )

    assert report.failures == []
    assert len(report.cases) == 1

    report_case = report.cases[0]

    assert report_case.assertions == {}
    assert len(report_case.evaluator_failures) == 1

    failure = report_case.evaluator_failures[0]

    assert failure.name == "response_required_states_order_status_delivered"
    assert failure.error_type == "RuntimeError"
    assert failure.error_message == "RuntimeError: judge unavailable"


def test_dataset_preserves_other_evaluation_results_when_response_judge_fails() -> None:
    def raise_judge_error(
        _messages: list[ModelMessage],
        _agent_info: AgentInfo,
    ) -> ModelResponse:
        raise RuntimeError("judge unavailable")

    successful_criterion = ResponseCriterion(
        id="states_order_status_delivered",
        statement="The response states that the order status is delivered.",
    )
    failed_criterion = ResponseCriterion(
        id="identifies_order_as_order_003",
        statement="The response identifies the order as order-003.",
    )

    dataset = Dataset[
        OrderEvalInput,
        OrderEvalOutput,
        OrderEvalMetadata,
    ](
        name="judge_failure_isolation",
        evaluators=(ToolSelectionEvaluator(),),
        cases=[
            Case(
                name="judge_failure_isolation",
                inputs=OrderEvalInput(
                    user_message="order-003은 지금 어떤 상태인가요?",
                    customer_id="customer-001",
                    execution_condition="default",
                ),
                metadata=OrderEvalMetadata(
                    scenario_id="scenario-1",
                    required_tool_uses=(
                        ExpectedToolUse(
                            tool_name="find_order",
                            expected_arguments={"order_id": "order-003"},
                            expected_outcome="success",
                        ),
                    ),
                    forbidden_tools=frozenset(),
                    required_tool_sequence=(),
                    required_response_criteria=(
                        successful_criterion,
                        failed_criterion,
                    ),
                    forbidden_response_criteria=(),
                ),
                evaluators=(
                    ResponseCriterionEvaluator(
                        criterion=successful_criterion,
                        criterion_kind="required",
                        judge_model=TestModel(
                            custom_output_args={
                                "reason": "The response states the delivered status.",
                                "pass": True,
                                "score": 1.0,
                            }
                        ),
                    ),
                    ResponseCriterionEvaluator(
                        criterion=failed_criterion,
                        criterion_kind="required",
                        judge_model=FunctionModel(raise_judge_error),
                    ),
                ),
            )
        ],
    )

    def return_agent_output(_inputs: OrderEvalInput) -> OrderEvalOutput:
        return OrderEvalOutput(
            agent_result=AgentResult(
                message="Order-003 has been delivered.",
            ),
            tool_uses=(
                ObservedToolUse(
                    tool_name="find_order",
                    arguments={"order_id": "order-003"},
                    outcome="success",
                ),
            ),
        )

    report = asyncio.run(
        dataset.evaluate(
            return_agent_output,
            progress=False,
        )
    )

    assert report.failures == []
    assert len(report.cases) == 1

    report_case = report.cases[0]

    assert set(report_case.assertions) == {
        "tool_selection",
        "response_required_states_order_status_delivered",
    }
    assert all(assertion.value is True for assertion in report_case.assertions.values())
    assert [failure.name for failure in report_case.evaluator_failures] == [
        "response_required_identifies_order_as_order_003"
    ]
