"""Verify the unknown-order flow with a real chat model."""

from _order_lookup import run_order_lookup_e2e

CUSTOMER_ID = "customer-001"
ORDER_ID = "order-999"
EXPECTED_TOOL_RESULT = {
    "error": {
        "code": "order_not_found",
        "message": "No order matched the provided order_id.",
    }
}


def main() -> int:
    return run_order_lookup_e2e(
        scenario_name="Order not found",
        customer_id=CUSTOMER_ID,
        order_id=ORDER_ID,
        expected_tool_result=EXPECTED_TOOL_RESULT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
