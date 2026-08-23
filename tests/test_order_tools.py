import pytest

from customer_support_agent.domain.fixtures import ORDERS
from customer_support_agent.tools.order import get_customer_orders


@pytest.mark.parametrize(
    ("customer_id", "expected_order_ids"),
    [
        ("customer-001", ["order-001", "order-002", "order-003"]),
        ("customer-002", ["order-004"]),
    ],
)
def test_get_customer_orders_returns_customer_orders_sorted_by_order_id(
    customer_id: str,
    expected_order_ids: list[str],
) -> None:
    actual_result = get_customer_orders(customer_id, {})

    expected_orders = [
        {
            "order_id": order_id,
            "status": ORDERS[order_id].status,
        }
        for order_id in expected_order_ids
    ]

    assert actual_result == {"orders": expected_orders}
