import logging
from typing import Annotated, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from customer_support_agent.domain.fixtures import ORDERS
from customer_support_agent.domain.models import OrderStatus
from customer_support_agent.tool_errors import (
    ToolError,
    create_tool_error,
)
from customer_support_agent.tools.definitions import ToolDefinition

logger = logging.getLogger(__name__)


class GetOrderSuccess(TypedDict):
    order_id: str
    status: OrderStatus


type GetOrderResult = GetOrderSuccess | ToolError


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
        logger.warning(
            "The tool arguments are invalid.",
        )
        return create_tool_error("invalid_arguments")

    logger.info(
        "The tool arguments are valid.",
    )

    order = ORDERS.get(parsed_arguments.order_id)

    if order is None:
        return create_tool_error("order_not_found")

    return {
        "order_id": order.order_id,
        "status": order.status,
    }
