from customer_support_agent.tools.customer_support import (
    CUSTOMER_SUPPORT_TOOLSET,
)
from customer_support_agent.tools.order import (
    find_order,
    get_customer_orders,
)
from customer_support_agent.tools.shipment import find_shipment


def test_customer_support_toolset_exposes_configured_tool_definitions() -> None:
    assert CUSTOMER_SUPPORT_TOOLSET.definitions == (
        get_customer_orders.definition,
        find_order.definition,
        find_shipment.definition,
    )
