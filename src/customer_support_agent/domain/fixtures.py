from collections.abc import Mapping
from types import MappingProxyType

from .models import (
    CancellationPolicy,
    Customer,
    Order,
    Shipment,
)

CUSTOMERS: Mapping[str, Customer] = MappingProxyType(
    {
        "customer-001": Customer(customer_id="customer-001"),
        "customer-002": Customer(customer_id="customer-002"),
    }
)

ORDERS: Mapping[str, Order] = MappingProxyType(
    {
        "order-001": Order(
            order_id="order-001", customer_id="customer-001", status="processing"
        ),
        "order-002": Order(
            order_id="order-002", customer_id="customer-001", status="shipped"
        ),
        "order-003": Order(
            order_id="order-003", customer_id="customer-001", status="delivered"
        ),
        "order-004": Order(
            order_id="order-004", customer_id="customer-002", status="cancelled"
        ),
    }
)

SHIPMENTS: Mapping[str, Shipment] = MappingProxyType(
    {
        "order-002": Shipment(order_id="order-002", status="out_for_delivery"),
    }
)

CANCELLATION_POLICY = CancellationPolicy(cancellable_statuses=frozenset({"processing"}))
