"""Run a manual E2E smoke test of the Agent -> get_order workflow."""

import logging
import os
import sys
from collections.abc import Sequence

from customer_support_agent.agent import Agent, AgentRunError
from customer_support_agent.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolResultPart,
)
from customer_support_agent.models import ChatModel, OpenAIChatModel
from customer_support_agent.tools.definitions import ToolDefinition

ORDER_ID = "order-002"
EXPECTED_STATUS = "shipped"


class RecordingChatModel(ChatModel):
    """에이전트가 get_order를 사용했는지 검증할 수 있도록 모델 요청 이력을 기록한다."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self.requests: list[tuple[ModelMessage, ...]] = []

    def generate(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        self.requests.append(tuple(messages))

        return self._model.generate(messages, tools)


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"{name} environment variable is required")

    return value.strip()


def require_expected_get_order_result(
    model: RecordingChatModel,
) -> None:
    expected_tool_result = {
        "order_id": ORDER_ID,
        "status": EXPECTED_STATUS,
    }
    tool_results = [
        part.result
        for message in model.requests[-1]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]

    if expected_tool_result not in tool_results:
        raise RuntimeError(
            f"Expected tool result {expected_tool_result!r}, got {tool_results!r}"
        )


def require_expected_final_answer(answer: str) -> None:
    expected_answer = f"RESULT: {ORDER_ID} | {EXPECTED_STATUS}"

    if answer.strip().casefold() != expected_answer.casefold():
        raise RuntimeError(f"Expected final answer {expected_answer!r}, got {answer!r}")


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-8s %(module)-10s %(message)s",
    )
    logging.getLogger("customer_support_agent").setLevel(logging.INFO)

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
    agent = Agent(model)

    try:
        answer = agent.run(
            f"Use the get_order tool to look up {ORDER_ID}. "
            "After the tool returns, respond with exactly one line "
            "in this format without backticks: "
            "RESULT: <order_id> | <status>. "
            "Replace the placeholders with the exact values from "
            "the tool result."
        )
    except AgentRunError as error:
        print(f"FAIL: {error.code}", file=sys.stderr)
        return 1

    require_expected_get_order_result(model)
    require_expected_final_answer(answer)

    print(f"PASS: order_id={ORDER_ID!r}, status={EXPECTED_STATUS!r}, answer={answer!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
