"""Shared execution and observation for order lookup E2E scenarios."""

import logging
import os
import sys
from collections.abc import Sequence
from textwrap import indent

from pydantic import BaseModel

from customer_support_agent.agent import AgentError, AgentResult
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

_EXPECTED_TOOL_NAME = "find_order"


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
        output_type: type[BaseModel] | None = None,
        instructions: str | None = None,
    ) -> ModelResponse:
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))

        return self._model.generate(
            messages, tools, output_type=output_type, instructions=instructions
        )


def _get_required_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"{name} environment variable is required")

    return value.strip()


def _require_customer_support_tool_definitions(model: RecordingChatModel) -> None:
    expected_definitions = CUSTOMER_SUPPORT_TOOLSET.definitions

    if len(model.tool_definitions) < 2:
        raise RuntimeError(
            "Expected at least one Tool-use call and one final response call."
        )

    for call_number, definitions in enumerate(model.tool_definitions[:-1], start=1):
        if definitions != expected_definitions:
            raise RuntimeError(
                "Expected all Customer Support Tool definitions on model call "
                f"{call_number}, got {definitions!r}"
            )

    if model.tool_definitions[-1]:
        raise RuntimeError(
            "Expected no Tool definitions on the final response call, got "
            f"{model.tool_definitions[-1]!r}"
        )


def _require_order_lookup_tool_flow(
    model: RecordingChatModel,
    *,
    order_id: str,
    expected_tool_result: object,
) -> None:
    if not model.requests:
        raise RuntimeError("The model was not called.")

    messages = model.requests[-1]
    tool_calls = [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    tool_results = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]
    expected_tool_arguments = {"order_id": order_id}

    for recorded_messages in model.requests:
        for tool_call_index, message in enumerate(recorded_messages):
            if not isinstance(message, ModelResponse):
                continue

            matching_tool_calls = (
                part
                for part in message.parts
                if isinstance(part, ToolCallPart)
                and part.name == _EXPECTED_TOOL_NAME
                and part.arguments == expected_tool_arguments
            )

            for tool_call in matching_tool_calls:
                for later_message in recorded_messages[tool_call_index + 1 :]:
                    if not isinstance(later_message, ModelRequest):
                        continue

                    if any(
                        isinstance(part, ToolResultPart)
                        and part.tool_call_id == tool_call.id
                        and part.result == expected_tool_result
                        for part in later_message.parts
                    ):
                        return

    raise RuntimeError(
        f"Expected {_EXPECTED_TOOL_NAME!r} with arguments "
        f"{expected_tool_arguments!r} and result {expected_tool_result!r}. "
        f"Observed Tool calls: {tool_calls!r}. "
        f"Observed Tool results: {tool_results!r}."
    )


def run_order_lookup_e2e(
    *,
    scenario_name: str,
    customer_id: str,
    order_id: str,
    expected_tool_result: object,
) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-8s %(module)-10s %(message)s",
    )
    logging.getLogger("customer_support_agent").setLevel(logging.INFO)
    result: AgentResult | None = None

    try:
        base_url = _get_required_env("LLM_BASE_URL")
        model_name = _get_required_env("LLM_MODEL_NAME")
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
            f"What is the current status of {order_id}?",
            context=ToolContext(customer_id=customer_id),
        )

        _require_customer_support_tool_definitions(model)
        _require_order_lookup_tool_flow(
            model,
            order_id=order_id,
            expected_tool_result=expected_tool_result,
        )
    except AgentError as error:
        print(f"FAIL: {error.code}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)

        if result is not None:
            print("Final answer:")
            print(indent(result.message, "  "))

        return 1

    print(f"PASS: {scenario_name} E2E verification succeeded.")
    print(f"Customer ID: {customer_id}")
    print(f"Order ID: {order_id}")
    print("Final answer:")
    print(indent(result.message, "  "))

    return 0
