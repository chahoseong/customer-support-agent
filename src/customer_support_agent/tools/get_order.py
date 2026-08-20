from typing import Annotated, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from customer_support_agent.tool_errors import (
    ToolError,
    create_tool_error,
)
from customer_support_agent.tools.definitions import ToolDefinition

type OrderStatus = Literal[
    "processing",
    "shipped",
    "delivered",
    "cancelled",
]


class GetOrderSuccess(TypedDict):
    order_id: str
    status: OrderStatus


type GetOrderResult = GetOrderSuccess | ToolError

_ORDERS: dict[str, OrderStatus] = {
    "order-001": "processing",
    "order-002": "shipped",
    "order-003": "delivered",
    "order-004": "cancelled",
}


class GetOrderArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    order_id: Annotated[
        str,
        Field(
            min_length=1,
            pattern=r"^\S(?:[\s\S]*\S)?$",
            description="Opaque order identifier to look up.",
        ),
    ]


GET_ORDER_TOOL_DEFINITION = ToolDefinition(
    name="get_order",
    description="Retrieve the current status of an order by its order_id.",
    parameters=GetOrderArguments.model_json_schema(),
)


def get_order(arguments: object) -> GetOrderResult:
    try:
        parsed_arguments = GetOrderArguments.model_validate(arguments)
    except ValidationError:
        return create_tool_error("invalid_arguments")

    status = _ORDERS.get(parsed_arguments.order_id)

    if status is None:
        return create_tool_error("order_not_found")

    return {
        "order_id": parsed_arguments.order_id,
        "status": status,
    }
