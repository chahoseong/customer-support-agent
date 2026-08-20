import pytest

import customer_support_agent.tools.get_order as get_order_module


@pytest.mark.parametrize(
    ("order_id", "expected_status"),
    [
        pytest.param("order-001", "processing", id="processing"),
        pytest.param("order-002", "shipped", id="shipped"),
        pytest.param("order-003", "delivered", id="delivered"),
        pytest.param("order-004", "cancelled", id="cancelled"),
    ],
)
def test_get_order_returns_known_order(
    order_id: str,
    expected_status: str,
) -> None:
    arguments = {"order_id": order_id}

    result = get_order_module.get_order(arguments)

    assert result == {
        "order_id": order_id,
        "status": expected_status,
    }


@pytest.mark.parametrize(
    "order_id",
    [
        pytest.param("order-999", id="unknown"),
    ],
)
def test_get_order_returns_not_found_error(
    order_id: str,
) -> None:
    arguments = {"order_id": order_id}

    result = get_order_module.get_order(arguments)

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
            {"order_id": "order-001", "unexpected": "value"},
            id="extra-field",
        ),
        pytest.param({"order_id": " order-001"}, id="leading-whitespace"),
        pytest.param({"order_id": "order-001 "}, id="trailing-whitespace"),
    ],
)
def test_get_order_returns_invalid_arguments_error(
    arguments: object,
) -> None:
    result = get_order_module.get_order(arguments)

    assert result == {
        "error": {
            "code": "invalid_arguments",
            "message": "Arguments do not match the tool's input schema.",
        }
    }


def test_get_order_arguments_schema_describes_input_contract() -> None:
    schema = get_order_module.GetOrderArguments.model_json_schema()
    order_id_schema = schema["properties"]["order_id"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["order_id"]
    assert order_id_schema["type"] == "string"
    assert order_id_schema["minLength"] == 1
    assert order_id_schema["pattern"] == r"^\S(?:[\s\S]*\S)?$"


def test_get_order_definition_describes_tool_contract() -> None:
    definition = get_order_module.GET_ORDER_TOOL_DEFINITION

    assert definition.name == "get_order"
    assert definition.description == (
        "Retrieve the current status of an order by its order_id."
    )
    assert (
        definition.parameters == get_order_module.GetOrderArguments.model_json_schema()
    )
