from typing import Annotated, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

type OrderStatus = Literal[
    "processing",
    "shipped",
    "delivered",
    "cancelled",
]

type GetOrderErrorCode = Literal[
    "invalid_arguments",
    "order_not_found",
]


class GetOrderSuccess(TypedDict):
    order_id: str
    status: OrderStatus


class GetOrderErrorDetails(TypedDict):
    code: GetOrderErrorCode
    message: str


class GetOrderError(TypedDict):
    error: GetOrderErrorDetails


type GetOrderResult = GetOrderSuccess | GetOrderError

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


def get_order(arguments: object) -> GetOrderResult:
    try:
        parsed_arguments = GetOrderArguments.model_validate(arguments)
    except ValidationError:
        return {
            "error": {
                "code": "invalid_arguments",
                "message": "Invalid arguments for get_order.",
            }
        }

    status = _ORDERS.get(parsed_arguments.order_id)

    if status is None:
        return {
            "error": {
                "code": "order_not_found",
                "message": "Order not found.",
            }
        }

    return {
        "order_id": parsed_arguments.order_id,
        "status": status,
    }
