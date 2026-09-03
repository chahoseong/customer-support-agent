from pathlib import Path

import pytest
from evals.order.dataset import load_order_dataset
from evals.order.evaluators import (
    ToolArgumentsEvaluator,
    ToolOutcomesEvaluator,
    ToolSelectionEvaluator,
    ToolTrajectoryEvaluator,
)
from evals.order.models import (
    ExpectedToolUse,
    OrderEvalInput,
    OrderEvalMetadata,
)
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("required_response_criteria", "forbidden_response_criteria"),
    [
        pytest.param(
            ("",),
            ("Do not report an unverified shipment status.",),
            id="empty-required-response-criterion",
        ),
        pytest.param(
            ("Report the verified shipment status.",),
            ("",),
            id="empty-forbidden-response-criterion",
        ),
    ],
)
def test_order_eval_metadata_rejects_empty_response_criterion(
    required_response_criteria: tuple[str, ...],
    forbidden_response_criteria: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        OrderEvalMetadata(
            scenario_id="scenario-3",
            required_tool_uses=(),
            forbidden_tools=frozenset(),
            required_tool_sequence=(),
            required_response_criteria=required_response_criteria,
            forbidden_response_criteria=forbidden_response_criteria,
        )


def test_load_order_dataset_returns_typed_cases_when_working_directory_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    dataset = load_order_dataset()

    assert dataset.name == "order_scenarios"
    assert dataset.cases
    assert all(isinstance(case.inputs, OrderEvalInput) for case in dataset.cases)
    assert all(isinstance(case.metadata, OrderEvalMetadata) for case in dataset.cases)


def test_load_order_dataset_returns_fresh_dataset_after_previous_load_is_modified() -> (
    None
):
    baseline = load_order_dataset()
    modified = load_order_dataset()
    source_case = modified.cases[0]

    modified.add_case(
        name="temporary_case",
        inputs=source_case.inputs,
        metadata=source_case.metadata,
    )

    reloaded = load_order_dataset()

    assert len(modified.cases) == len(baseline.cases) + 1
    assert modified is not reloaded
    assert reloaded.cases == baseline.cases


def test_expected_tool_use_preserves_tool_name_arguments_and_outcome() -> None:
    expected_tool_use = ExpectedToolUse(
        tool_name="find_order",
        expected_arguments={"order_id": "order-002"},
        expected_outcome="success",
    )

    assert (
        expected_tool_use.tool_name,
        expected_tool_use.expected_arguments,
        expected_tool_use.expected_outcome,
    ) == (
        "find_order",
        {"order_id": "order-002"},
        "success",
    )


def test_order_eval_metadata_preserves_tool_use_requirements() -> None:
    required_tool_use = ExpectedToolUse(
        tool_name="lookup_subject",
        expected_arguments={"subject_id": "subject-001"},
        expected_outcome="success",
    )

    metadata = OrderEvalMetadata(
        scenario_id="scenario-1",
        required_tool_uses=(required_tool_use,),
        forbidden_tools=frozenset({"lookup_unrelated"}),
        required_tool_sequence=(),
        required_response_criteria=("State the verified result.",),
        forbidden_response_criteria=("Do not state unverified facts.",),
    )

    assert (
        metadata.required_tool_uses,
        metadata.forbidden_tools,
        metadata.required_tool_sequence,
    ) == (
        (required_tool_use,),
        frozenset({"lookup_unrelated"}),
        (),
    )


def test_order_eval_metadata_rejects_duplicate_required_tool_names() -> None:
    with pytest.raises(
        ValidationError,
        match="required tool names must be unique",
    ):
        OrderEvalMetadata(
            scenario_id="scenario-1",
            required_tool_uses=(
                ExpectedToolUse(
                    tool_name="lookup_subject",
                    expected_arguments={"subject_id": "subject-001"},
                    expected_outcome="success",
                ),
                ExpectedToolUse(
                    tool_name="lookup_subject",
                    expected_arguments={"subject_id": "subject-002"},
                    expected_outcome="success",
                ),
            ),
            forbidden_tools=frozenset(),
            required_tool_sequence=(),
            required_response_criteria=("State the verified result.",),
            forbidden_response_criteria=("Do not state unverified facts.",),
        )


def test_order_eval_metadata_rejects_overlapping_required_and_forbidden_tools() -> None:
    with pytest.raises(
        ValidationError,
        match="required and forbidden tools must not overlap",
    ):
        OrderEvalMetadata(
            scenario_id="scenario-1",
            required_tool_uses=(
                ExpectedToolUse(
                    tool_name="lookup_subject",
                    expected_arguments={"subject_id": "subject-001"},
                    expected_outcome="success",
                ),
            ),
            forbidden_tools=frozenset({"lookup_subject"}),
            required_tool_sequence=(),
            required_response_criteria=("State the verified result.",),
            forbidden_response_criteria=("Do not state unverified facts.",),
        )


def test_order_eval_metadata_rejects_sequence_that_references_non_required_tool() -> (
    None
):
    with pytest.raises(
        ValidationError,
        match="required tool sequence must reference only required tools",
    ):
        OrderEvalMetadata(
            scenario_id="scenario-1",
            required_tool_uses=(
                ExpectedToolUse(
                    tool_name="lookup_subject",
                    expected_arguments={"subject_id": "subject-001"},
                    expected_outcome="success",
                ),
            ),
            forbidden_tools=frozenset(),
            required_tool_sequence=("lookup_subject", "lookup_detail"),
            required_response_criteria=("State the verified result.",),
            forbidden_response_criteria=("Do not state unverified facts.",),
        )


def test_order_eval_metadata_rejects_duplicate_tool_names_in_required_sequence() -> (
    None
):
    with pytest.raises(
        ValidationError,
        match="required tool sequence must not contain duplicate tool names",
    ):
        OrderEvalMetadata(
            scenario_id="scenario-1",
            required_tool_uses=(
                ExpectedToolUse(
                    tool_name="lookup_subject",
                    expected_arguments={"subject_id": "subject-001"},
                    expected_outcome="success",
                ),
                ExpectedToolUse(
                    tool_name="lookup_detail",
                    expected_arguments={"subject_id": "subject-001"},
                    expected_outcome="success",
                ),
            ),
            forbidden_tools=frozenset(),
            required_tool_sequence=(
                "lookup_subject",
                "lookup_detail",
                "lookup_subject",
            ),
            required_response_criteria=("State the verified result.",),
            forbidden_response_criteria=("Do not state unverified facts.",),
        )


def test_load_order_dataset_returns_all_tool_evaluators_in_stable_order() -> None:
    dataset = load_order_dataset()

    assert tuple(type(evaluator) for evaluator in dataset.evaluators) == (
        ToolSelectionEvaluator,
        ToolArgumentsEvaluator,
        ToolOutcomesEvaluator,
        ToolTrajectoryEvaluator,
    )


def test_load_order_dataset_returns_fresh_tool_evaluator_instances() -> None:
    first_dataset = load_order_dataset()
    second_dataset = load_order_dataset()

    assert all(
        first_evaluator is not second_evaluator
        for first_evaluator, second_evaluator in zip(
            first_dataset.evaluators,
            second_dataset.evaluators,
            strict=True,
        )
    )
