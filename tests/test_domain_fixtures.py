from collections.abc import Mapping, MutableMapping
from typing import cast

import pytest

from customer_support_agent.domain.fixtures import (
    CANCELLATION_POLICY,
    CUSTOMERS,
    ORDERS,
    SHIPMENTS,
)
from customer_support_agent.domain.models import OrderStatus


@pytest.mark.parametrize(
    ("records", "expected_keys"),
    [
        (CUSTOMERS, {"customer-001", "customer-002"}),
        (ORDERS, {"order-001", "order-002", "order-003", "order-004"}),
        (SHIPMENTS, {"order-002"}),
    ],
)
def test_canonical_fixtures_contain_only_approved_records(
    records: Mapping[str, object],
    expected_keys: set[str],
) -> None:
    assert set(records) == expected_keys


def test_cancellation_policy_contains_only_processing_status() -> None:
    assert CANCELLATION_POLICY.cancellable_statuses == frozenset({"processing"})


@pytest.mark.parametrize("records", [CUSTOMERS, ORDERS, SHIPMENTS])
def test_canonical_fixture_mappings_reject_item_assignment(
    records: Mapping[str, object],
) -> None:
    unexpected_key = "unexpected"
    mutable_records = cast(MutableMapping[str, object], records)

    try:
        with pytest.raises(TypeError):
            mutable_records[unexpected_key] = object()
    finally:
        if isinstance(records, dict):
            records.pop(unexpected_key, None)


@pytest.mark.parametrize(
    ("records", "identifier_attribute"),
    [
        (CUSTOMERS, "customer_id"),
        (ORDERS, "order_id"),
        (SHIPMENTS, "order_id"),
    ],
)
def test_fixture_mapping_keys_match_record_identifiers(
    records: Mapping[str, object],
    identifier_attribute: str,
) -> None:
    assert all(
        key == getattr(record, identifier_attribute) for key, record in records.items()
    )


@pytest.mark.parametrize(
    ("customer_id", "expected_order_ids"),
    [
        ("customer-001", {"order-001", "order-002", "order-003"}),
        ("customer-002", {"order-004"}),
    ],
)
def test_canonical_orders_preserve_scenario_ownership(
    customer_id: str, expected_order_ids: set[str]
) -> None:
    actual_order_ids = {
        order.order_id for order in ORDERS.values() if order.customer_id == customer_id
    }

    assert actual_order_ids == expected_order_ids


@pytest.mark.parametrize(
    ("order_id", "expected_status"),
    [
        ("order-001", "processing"),
        ("order-002", "shipped"),
        ("order-003", "delivered"),
    ],
)
def test_canonical_orders_have_statuses_required_by_support_scenarios(
    order_id: str, expected_status: OrderStatus
) -> None:
    assert ORDERS[order_id].status == expected_status


def test_canonical_shipment_has_status_required_by_support_scenario() -> None:
    assert SHIPMENTS["order-002"].status == "out_for_delivery"
