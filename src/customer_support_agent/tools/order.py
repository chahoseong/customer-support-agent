from typing import TypedDict

from pydantic import BaseModel, ConfigDict, ValidationError

from customer_support_agent.domain.fixtures import ORDERS
from customer_support_agent.domain.models import Order, OrderStatus
from customer_support_agent.tool_errors import ToolError, create_tool_error


class CustomerOrderSummary(TypedDict):
    order_id: str
    status: OrderStatus


class GetCustomerOrdersSuccess(TypedDict):
    orders: list[CustomerOrderSummary]


type GetCustomerOrdersResult = GetCustomerOrdersSuccess | ToolError


class GetCustomerOrdersArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


def get_customer_orders(
    customer_id: str,
    arguments: object,
) -> GetCustomerOrdersResult:
    try:
        GetCustomerOrdersArguments.model_validate(arguments)
    except ValidationError:
        return create_tool_error("invalid_arguments")

    orders: list[Order] = []

    for order in ORDERS.values():
        if customer_id == order.customer_id:
            orders.append(order)

    orders.sort(key=lambda order: order.order_id)

    return {
        "orders": [
            {
                "order_id": order.order_id,
                "status": order.status,
            }
            for order in orders
        ]
    }
