from pathlib import Path

import pytest
from evals.order.dataset import load_order_dataset
from evals.order.models import (
    InformationSource,
    InformationSourceExpectation,
    OrderEvalInput,
    OrderEvalMetadata,
)
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("source", "order_id"),
    [
        pytest.param("order", None, id="order-without-order-id"),
        pytest.param("shipment", None, id="shipment-without-order-id"),
        pytest.param(
            "customer_orders",
            "order-001",
            id="customer-orders-with-order-id",
        ),
        pytest.param(
            "cancellation_policy",
            "order-001",
            id="cancellation-policy-with-order-id",
        ),
    ],
)
def test_information_source_expectation_rejects_order_id_that_conflicts_with_source(
    source: InformationSource,
    order_id: str | None,
) -> None:
    with pytest.raises(ValidationError):
        InformationSourceExpectation(
            source=source,
            order_id=order_id,
            outcome="available",
        )


@pytest.mark.parametrize(
    ("source", "order_id"),
    [
        pytest.param("customer_orders", None, id="customer-orders-without-order-id"),
        pytest.param(
            "cancellation_policy",
            None,
            id="cancellation-policy-without-order-id",
        ),
        pytest.param("order", "order-001", id="order-with-order-id"),
        pytest.param("shipment", "order-001", id="shipment-with-order-id"),
    ],
)
def test_information_source_expectation_accepts_order_id_that_matches_source(
    source: InformationSource,
    order_id: str | None,
) -> None:
    expectation = InformationSourceExpectation(
        source=source, order_id=order_id, outcome="available"
    )

    assert expectation.source == source
    assert expectation.order_id == order_id


def test_order_eval_metadata_rejects_overlapping_required_and_forbidden_information_sources() -> (
    None
):
    with pytest.raises(ValidationError):
        OrderEvalMetadata(
            scenario_id="scenario-3",
            required_information_sources=frozenset(
                {
                    InformationSourceExpectation(
                        source="shipment",
                        order_id="order-002",
                        outcome="available",
                    )
                }
            ),
            forbidden_information_sources=frozenset({"shipment"}),
            required_response_criteria=("Report the verified shipment status.",),
            forbidden_response_criteria=(
                "Do not report an unverified shipment status.",
            ),
        )


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
            required_information_sources=frozenset(
                {
                    InformationSourceExpectation(
                        source="shipment",
                        order_id="order-002",
                        outcome="available",
                    )
                }
            ),
            forbidden_information_sources=frozenset({"customer_orders"}),
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


def test_information_source_expectation_rejects_empty_order_id() -> None:
    with pytest.raises(ValidationError):
        InformationSourceExpectation(source="order", order_id="", outcome="available")
