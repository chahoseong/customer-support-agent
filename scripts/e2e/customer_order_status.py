"""Verify the customer order status flow with a real chat model."""

import logging
import os
import sys
from collections.abc import Sequence
from textwrap import indent

from pydantic import BaseModel

from customer_support_agent.agent import AgentError
from customer_support_agent.customer_support import (
    CUSTOMER_SUPPORT_TOOLSET,
    create_customer_support_agent,
)
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolResultPart,
)
from customer_support_agent.models import ChatModel, OpenAIChatModel
from customer_support_agent.tools import ToolContext, ToolDefinition

CUSTOMER_ID = "customer-001"
ORDER_ID = "order-002"

EXPECTED_TOOL_RESULTS: tuple[tuple[str, object], ...] = (
    (
        "get_customer_orders",
        {
            "orders": [
                {"order_id": "order-001", "status": "processing"},
                {"order_id": "order-002", "status": "shipped"},
                {"order_id": "order-003", "status": "delivered"},
            ]
        },
    ),
    (
        "find_order",
        {"order_id": ORDER_ID, "status": "shipped"},
    ),
    (
        "find_shipment",
        {"order_id": ORDER_ID, "shipment_status": "out_for_delivery"},
    ),
)


class RecordingChatModel(ChatModel):
    """Record each request and Tool definition set sent to the real model."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self.requests: list[tuple[ModelMessage, ...]] = []
        self.tool_definitions: list[tuple[ToolDefinition, ...]] = []

    def generate(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        *,
        output_type: type[BaseModel],
        instructions: str | None = None,
    ) -> ModelResponse:
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))

        return self._model.generate(
            messages, tools, output_type=output_type, instructions=instructions
        )


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"{name} environment variable is required")

    return value.strip()


def require_customer_support_tool_definitions(model: RecordingChatModel) -> None:
    expected_definitions = CUSTOMER_SUPPORT_TOOLSET.definitions

    if not model.tool_definitions:
        raise RuntimeError("The model was not called.")

    for call_number, definitions in enumerate(model.tool_definitions, start=1):
        if definitions != expected_definitions:
            raise RuntimeError(
                "Expected all Customer Support Tool definitions on model call "
                f"{call_number}, got {definitions!r}"
            )


def require_customer_order_status_tool_flow(model: RecordingChatModel) -> None:
    if not model.requests:
        raise RuntimeError("The model was not called.")

    messages = model.requests[-1]
    tool_calls: list[ToolCallPart] = [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    expected_tool_names = [name for name, _ in EXPECTED_TOOL_RESULTS]
    actual_tool_names = [tool_call.name for tool_call in tool_calls]

    if actual_tool_names != expected_tool_names:
        raise RuntimeError(
            f"Expected Tool calls {expected_tool_names!r}, got {actual_tool_names!r}"
        )

    tool_results = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]

    if len(tool_results) != len(tool_calls):
        raise RuntimeError(
            f"Expected {len(tool_calls)} Tool results, got {len(tool_results)}"
        )

    for tool_call, tool_result, (_, expected_result) in zip(
        tool_calls,
        tool_results,
        EXPECTED_TOOL_RESULTS,
        strict=True,
    ):
        if tool_result.tool_call_id != tool_call.id:
            raise RuntimeError(
                f"Expected result for Tool Call ID {tool_call.id!r}, "
                f"got {tool_result.tool_call_id!r}"
            )

        if tool_result.result != expected_result:
            raise RuntimeError(
                f"Expected result {expected_result!r} from {tool_call.name!r}, "
                f"got {tool_result.result!r}"
            )


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-8s %(module)-10s %(message)s",
    )
    logging.getLogger("customer_support_agent").setLevel(logging.INFO)

    try:
        base_url = get_required_env("LLM_BASE_URL")
        model_name = get_required_env("LLM_MODEL_NAME")
        api_key = os.getenv("LLM_API_KEY", "").strip() or "no-api-key"

        model = RecordingChatModel(
            OpenAIChatModel(
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
            )
        )
        agent = create_customer_support_agent(model)
        result = agent.run(
            "Call the available tools in exactly this order: "
            "get_customer_orders with no arguments, "
            f"find_order for {ORDER_ID}, and then find_shipment for {ORDER_ID}. "
            "After all three tools return, summarize every order_id and status from "
            "get_customer_orders, the status from find_order, and the shipment_status "
            "from find_shipment. Preserve the exact identifier and status values from "
            "the Tool results in your final answer.",
            context=ToolContext(customer_id=CUSTOMER_ID),
        )

        require_customer_support_tool_definitions(model)
        require_customer_order_status_tool_flow(model)
    except AgentError as error:
        print(f"FAIL: {error.code}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: Customer order status E2E verification succeeded.")
    print(f"Customer ID: {CUSTOMER_ID}")
    print("Final answer:")
    print(indent(result.message, "  "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
