from .order import find_order, get_customer_orders
from .policy import get_cancellation_policy
from .shipment import find_shipment
from .toolset import Toolset

CUSTOMER_SUPPORT_TOOLSET = Toolset(
    tools=(
        get_customer_orders,
        find_order,
        find_shipment,
        get_cancellation_policy,
    )
)
