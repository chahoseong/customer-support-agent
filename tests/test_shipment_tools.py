import pytest

from customer_support_agent.tools.shipment import (
    FIND_SHIPMENT_TOOL_DEFINITION,
    FindShipmentArguments,
    find_shipment,
)


def test_find_shipment_returns_shipment_in_customer_scope() -> None:
    result = find_shipment("customer-001", {"order_id": "order-002"})

    assert result == {
        "order_id": "order-002",
        "shipment_status": "out_for_delivery",
    }


@pytest.mark.parametrize(
    ("customer_id", "order_id"),
    [
        pytest.param("customer-001", "order-001", id="pre-shipment-order"),
        pytest.param("customer-001", "order-003", id="delivered-order"),
        pytest.param("customer-001", "order-999", id="unknown-order"),
        pytest.param("customer-002", "order-002", id="other-customer-order"),
    ],
)
def test_find_shipment_returns_not_found_when_shipment_is_unavailable_to_customer(
    customer_id: str,
    order_id: str,
) -> None:
    result = find_shipment(customer_id, {"order_id": order_id})

    assert result == {
        "error": {
            "code": "shipment_not_found",
            "message": (
                "No shipment information is available for the provided order_id."
            ),
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
            {"order_id": "order-002", "unexpected": "value"},
            id="extra-field",
        ),
        pytest.param({"order_id": " order-002"}, id="leading-whitespace"),
        pytest.param({"order_id": "order-002 "}, id="trailing-whitespace"),
        pytest.param([], id="non-object"),
    ],
)
def test_find_shipment_returns_invalid_arguments_error(arguments: object) -> None:
    result = find_shipment("customer-001", arguments)

    assert result == {
        "error": {
            "code": "invalid_arguments",
            "message": "Arguments do not match the tool's input schema.",
        }
    }


def test_find_shipment_arguments_schema_describes_input_contract() -> None:
    schema = FindShipmentArguments.model_json_schema()
    order_id_schema = schema["properties"]["order_id"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["order_id"]
    assert order_id_schema["type"] == "string"
    assert order_id_schema["minLength"] == 1
    assert order_id_schema["pattern"] == r"^\S(?:[\s\S]*\S)?$"


def test_find_shipment_definition_describes_conditional_customer_scoped_lookup() -> (
    None
):
    definition = FIND_SHIPMENT_TOOL_DEFINITION

    assert definition.name == "find_shipment"
    assert definition.description == (
        "Retrieve detailed shipment information for an order belonging to "
        "the current customer after an order lookup indicates it is needed. "
        "Returns order_id and shipment_status, or shipment_not_found when no "
        "shipment information is available."
    )
    assert definition.parameters == FindShipmentArguments.model_json_schema()
