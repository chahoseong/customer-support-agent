from typing import Annotated, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from customer_support_agent.domain.fixtures import (
    ORDERS,
    SHIPMENTS,
)
from customer_support_agent.domain.models import (
    ShipmentStatus,
)

from .errors import (
    ToolError,
    create_tool_error,
)
from .tool import ToolContext, tool


class FindShipmentSuccess(TypedDict):
    order_id: str
    shipment_status: ShipmentStatus


type FindShipmentResult = FindShipmentSuccess | ToolError


class FindShipmentArguments(BaseModel):
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
def find_shipment(
    context: ToolContext,
    arguments: FindShipmentArguments,
) -> FindShipmentResult:
    """Retrieve detailed shipment information for an order belonging to the current customer after an order lookup indicates it is needed. Returns order_id and shipment_status, or shipment_not_found when no shipment information is available."""
    order_id = arguments.order_id

    order = ORDERS.get(order_id)
    if order is None or order.customer_id != context.customer_id:
        return create_tool_error("shipment_not_found")

    shipment = SHIPMENTS.get(order_id)
    if shipment is None:
        return create_tool_error("shipment_not_found")

    return {
        "order_id": shipment.order_id,
        "shipment_status": shipment.status,
    }
