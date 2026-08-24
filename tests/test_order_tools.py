import pytest

import customer_support_agent.tools.order as order_module
from customer_support_agent.domain.fixtures import ORDERS
from customer_support_agent.tools.order import (
    FIND_ORDER_TOOL_DEFINITION,
    GET_CUSTOMER_ORDERS_TOOL_DEFINITION,
    FindOrderArguments,
    GetCustomerOrdersArguments,
    find_order,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reversed_orders = dict(reversed(tuple(ORDERS.items())))
    monkeypatch.setattr(order_module, "ORDERS", reversed_orders)

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


@pytest.mark.parametrize(
    ("customer_id", "order_id"),
    [
        ("customer-001", "order-001"),
        ("customer-002", "order-004"),
    ],
)
def test_find_order_returns_order_in_customer_scope(
    customer_id: str,
    order_id: str,
) -> None:
    result = find_order(customer_id, {"order_id": order_id})

    assert result == {
        "order_id": order_id,
        "status": ORDERS[order_id].status,
    }


@pytest.mark.parametrize(
    ("customer_id", "order_id"),
    [
        pytest.param("customer-001", "order-999", id="unknown-order"),
        pytest.param("customer-001", "order-004", id="other-customer-order"),
    ],
)
def test_find_order_returns_not_found_when_order_is_not_in_customer_scope(
    customer_id: str,
    order_id: str,
) -> None:
    result = find_order(customer_id, {"order_id": order_id})

    assert result == {
        "error": {
            "code": "order_not_found",
            "message": "No order matched the provided order_id.",
        }
    }


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({}, id="missing"),
        pytest.param({"order_id": 123}, id="wrong-type"),
        pytest.param({"order_id": ""}, id="empty"),
        pytest.param({"order_id": "   "}, id="whitespace-only"),
        pytest.param(
            {"order_id": "order-001", "unexpected": "value"}, id="extra-field"
        ),
        pytest.param({"order_id": " order-001"}, id="leading-whitespace"),
        pytest.param({"order_id": "order-001 "}, id="trailing-whitespace"),
        pytest.param([], id="non-object"),
    ],
)
def test_find_order_returns_invalid_arguments_error(arguments: object) -> None:
    result = find_order("customer-001", arguments)

    assert result == {
        "error": {
            "code": "invalid_arguments",
            "message": "Arguments do not match the tool's input schema.",
        }
    }


def test_find_order_arguments_schema_describes_input_contract() -> None:
    schema = FindOrderArguments.model_json_schema()
    order_id_schema = schema["properties"]["order_id"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["order_id"]
    assert order_id_schema["type"] == "string"
    assert order_id_schema["minLength"] == 1
    assert order_id_schema["pattern"] == r"^\S(?:[\s\S]*\S)?$"


def test_find_order_definition_describes_customer_scoped_lookup() -> None:
    definition = FIND_ORDER_TOOL_DEFINITION

    assert definition.name == "find_order"
    assert definition.description == (
        "Retrieve the current status of an order belonging to the current "
        "customer by its order_id."
    )
    assert definition.parameters == FindOrderArguments.model_json_schema()
