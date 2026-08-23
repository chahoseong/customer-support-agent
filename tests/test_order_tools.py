import pytest

from customer_support_agent.domain.fixtures import ORDERS
from customer_support_agent.tools.order import (
    GET_CUSTOMER_ORDERS_TOOL_DEFINITION,
    GetCustomerOrdersArguments,
    get_customer_orders,
)


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


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"unexpected": "value"}, id="unexpected-field"),
        pytest.param([], id="non-object"),
    ],
)
def test_get_customer_orders_returns_invalid_arguments_error(arguments: object) -> None:

    result = get_customer_orders("customer-001", arguments)

    assert result == {
        "error": {
            "code": "invalid_arguments",
            "message": "Arguments do not match the tool's input schema.",
        }
    }


def test_get_customer_orders_arguments_schema_describes_empty_object_contract() -> None:
    schema = GetCustomerOrdersArguments.model_json_schema()

    assert schema["type"] == "object"
    assert schema["properties"] == {}
    assert schema["additionalProperties"] is False


def test_get_customer_orders_definition_describes_customer_scoped_lookup() -> None:
    definition = GET_CUSTOMER_ORDERS_TOOL_DEFINITION

    assert definition.name == "get_customer_orders"
    assert definition.description == (
        "Retrieve the current customer's orders, including each order's "
        "order_id and current status."
    )
    assert definition.parameters == GetCustomerOrdersArguments.model_json_schema()
