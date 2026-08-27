"""Verify the known-order status flow with a real chat model."""

from _order_lookup import run_order_lookup_e2e

CUSTOMER_ID = "customer-001"
ORDER_ID = "order-001"
EXPECTED_TOOL_RESULT = {
    "order_id": ORDER_ID,
    "status": "processing",
}


def main() -> int:
    return run_order_lookup_e2e(
        scenario_name="Order status",
        customer_id=CUSTOMER_ID,
        order_id=ORDER_ID,
        expected_tool_result=EXPECTED_TOOL_RESULT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
