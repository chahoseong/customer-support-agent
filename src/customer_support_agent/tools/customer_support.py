from .order import find_order, get_customer_orders
from .shipment import find_shipment
from .toolset import Toolset

CUSTOMER_SUPPORT_TOOLSET = Toolset(
    tools=(
        get_customer_orders,
        find_order,
        find_shipment,
    )
)
