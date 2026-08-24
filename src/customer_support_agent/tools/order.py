from typing import Annotated, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from customer_support_agent.domain.fixtures import ORDERS
from customer_support_agent.domain.models import Order, OrderStatus
from customer_support_agent.tools.errors import ToolError, create_tool_error

from .tool import ToolContext, tool


class CustomerOrderSummary(TypedDict):
    order_id: str
    status: OrderStatus


class GetCustomerOrdersSuccess(TypedDict):
    orders: list[CustomerOrderSummary]


type GetCustomerOrdersResult = GetCustomerOrdersSuccess | ToolError


class GetCustomerOrdersArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


@tool
def get_customer_orders(
    context: ToolContext,
    arguments: GetCustomerOrdersArguments,
) -> GetCustomerOrdersResult:
    """Retrieve the current customer's orders, including each order's order_id and current status."""

    orders: list[Order] = []

    for order in ORDERS.values():
        if context.customer_id == order.customer_id:
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


class FindOrderSuccess(TypedDict):
    order_id: str
    status: OrderStatus


type FindOrderResult = FindOrderSuccess | ToolError


class FindOrderArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    order_id: Annotated[
        str,
        Field(
            min_length=1,
            pattern=r"^\S(?:[\s\S]*\S)?$",
            description="Opaque order identifier to look up.",
        ),
    ]


@tool
def find_order(
    context: ToolContext,
    arguments: FindOrderArguments,
) -> FindOrderResult:
    """Retrieve the current status of an order belonging to the current customer by its order_id."""
    order = ORDERS.get(arguments.order_id)

    if order is None or order.customer_id != context.customer_id:
        return create_tool_error("order_not_found")

    return {"order_id": order.order_id, "status": order.status}
