from typing import TypedDict

from customer_support_agent.domain.fixtures import CANCELLATION_POLICY
from customer_support_agent.domain.models import OrderStatus

from .tool import tool


class GetCancellationPolicySuccess(TypedDict):
    cancellable_statuses: list[OrderStatus]


@tool
def get_cancellation_policy() -> GetCancellationPolicySuccess:
    """Retrieve the global cancellation policy as the order statuses that permit cancellation."""
    return {"cancellable_statuses": sorted(CANCELLATION_POLICY.cancellable_statuses)}
