from dataclasses import dataclass
from typing import Literal

type OrderStatus = Literal[
    "processing",
    "shipped",
    "delivered",
    "cancelled",
]

type ShipmentStatus = Literal["out_for_delivery",]


@dataclass(frozen=True)
class Customer:
    customer_id: str


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    status: OrderStatus


@dataclass(frozen=True)
class Shipment:
    order_id: str
    status: ShipmentStatus


@dataclass(frozen=True)
class CancellationPolicy:
    cancellable_statuses: frozenset[OrderStatus]
