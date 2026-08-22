import pytest

from customer_support_agent.domain.models import (
    CancellationPolicy,
    Customer,
    Order,
    Shipment,
)


@pytest.mark.parametrize(
    ("record", "attribute", "new_value"),
    [
        (
            Customer(
                customer_id="customer-001",
            ),
            "customer_id",
            "customer-002",
        ),
        (
            Order(
                order_id="order-001",
                customer_id="customer-001",
                status="processing",
            ),
            "status",
            "shipped",
        ),
        (
            Shipment(
                order_id="order-002",
                status="out_for_delivery",
            ),
            "status",
            "unknown",
        ),
        (
            CancellationPolicy(
                cancellable_statuses=frozenset({"processing"}),
            ),
            "cancellable_statuses",
            frozenset(),
        ),
    ],
)
def test_domain_data_objects_reject_attribute_reassignment(
    record: object,
    attribute: str,
    new_value: object,
) -> None:
    with pytest.raises(AttributeError):
        setattr(record, attribute, new_value)
