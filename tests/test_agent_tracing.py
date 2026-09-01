import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest
from logfire.testing import CaptureLogfire
from logfire.types import SpanLevel
from pydantic import BaseModel

from customer_support_agent.agent import (
    Agent,
    AgentError,
    AgentResult,
)
from customer_support_agent.messages import (
    ModelMessage,
    ModelResponse,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
)
from customer_support_agent.models import ChatModel
from customer_support_agent.tools import (
    ToolContext,
    ToolDefinition,
    Toolset,
    tool,
)
from customer_support_agent.tools.errors import create_tool_error

from .scripted_model import ScriptedModel

type _ExportedSpan = dict[str, Any]

_PROJECT_SPAN_NAMES = frozenset(
    {
        "agent.run",
        "model.generate",
        "tool.execute",
    }
)

_SENSITIVE_MAPPING_ERROR = "sensitive mapping failure"
_SENSITIVE_STRING_ERROR = "sensitive string failure"


class _HostileMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError(_SENSITIVE_MAPPING_ERROR)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError(_SENSITIVE_MAPPING_ERROR)

    def __len__(self) -> int:
        return 1


class _HostileComparableString(str):
    def __lt__(self, value: str, /) -> bool:
        raise RuntimeError(_SENSITIVE_STRING_ERROR)


class _HostileHashString(str):
    def __hash__(self) -> int:
        raise RuntimeError(_SENSITIVE_STRING_ERROR)


def _get_project_spans(capfire: CaptureLogfire) -> list[_ExportedSpan]:
    exported_spans = capfire.exporter.exported_spans_as_dict(
        parse_json_attributes=True,
    )

    return [span for span in exported_spans if span["name"] in _PROJECT_SPAN_NAMES]


def _assert_is_direct_child_of(
    child_span: _ExportedSpan,
    parent_span: _ExportedSpan,
) -> None:
    assert child_span["context"]["trace_id"] == parent_span["context"]["trace_id"]
    assert child_span["parent"] is not None
    assert child_span["parent"]["span_id"] == parent_span["context"]["span_id"]


def test_agent_trace_contains_root_and_model_children_without_tool_calls(
    capfire: CaptureLogfire,
) -> None:
    expected_result = AgentResult(
        message="Your order is processing",
    )

    model = ScriptedModel(
        [
            ModelResponse(parts=(TextPart(content=expected_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    agent = Agent(model, toolset=Toolset(tools=()))
    agent.run("Where is my order?", context=ToolContext(customer_id="customer-001"))

    project_spans = _get_project_spans(capfire)

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]

    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 2

    root_span = root_spans[0]
    model_spans.sort(key=lambda span: span["start_time"])

    assert root_span["parent"] is None

    root_attributes = root_span["attributes"]

    assert root_attributes["customer_support_agent.agent.outcome"] == "success"
    assert root_attributes["customer_support_agent.agent.model_call.count"] == 2
    assert root_attributes["customer_support_agent.agent.tool_call.count"] == 0

    for model_span in model_spans:
        _assert_is_direct_child_of(model_span, root_span)

    model_attributes = [model_span["attributes"] for model_span in model_spans]

    assert [
        attributes["customer_support_agent.model.call.index"]
        for attributes in model_attributes
    ] == [1, 2]

    assert [
        attributes["customer_support_agent.model.call.purpose"]
        for attributes in model_attributes
    ] == ["agent_loop", "final_result"]

    assert [
        attributes["customer_support_agent.model.outcome"]
        for attributes in model_attributes
    ] == ["success", "success"]

    assert [
        attributes["customer_support_agent.model.response.kind"]
        for attributes in model_attributes
    ] == ["text", "agent_result"]

    assert [
        attributes["customer_support_agent.model.tool_call.count"]
        for attributes in model_attributes
    ] == [0, 0]


def test_agent_trace_separates_consecutive_runs(
    capfire: CaptureLogfire,
) -> None:
    first_result = AgentResult(message="Your first order is processing")
    second_result = AgentResult(message="Your second order has shipped")

    model = ScriptedModel(
        [
            ModelResponse(parts=(TextPart(content=first_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=first_result),)),
            ModelResponse(parts=(TextPart(content=second_result.message),)),
            ModelResponse(parts=(StructuredOutputPart(output=second_result),)),
        ]
    )

    agent = Agent(model, toolset=Toolset(tools=()))
    context = ToolContext(customer_id="customer-001")

    agent.run("Where is my first order?", context=context)
    agent.run("Where is my second order?", context=context)

    project_spans = _get_project_spans(capfire)
    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 2
    assert len(model_spans) == 4

    root_spans.sort(key=lambda span: span["start_time"])
    root_trace_ids = [span["context"]["trace_id"] for span in root_spans]

    assert root_trace_ids[0] != root_trace_ids[1]

    for root_span in root_spans:
        assert root_span["parent"] is None

        root_context = root_span["context"]
        child_model_spans = [
            span
            for span in model_spans
            if span["context"]["trace_id"] == root_context["trace_id"]
        ]

        assert len(child_model_spans) == 2

        for model_span in child_model_spans:
            _assert_is_direct_child_of(model_span, root_span)


def test_agent_trace_records_model_exception_without_sensitive_details(
    capfire: CaptureLogfire,
) -> None:
    model_error = RuntimeError("sensitive provider failure")

    class FailingModel(ChatModel):
        def generate(
            self,
            _messages: Sequence[ModelMessage],
            tools: Sequence[ToolDefinition],
            *,
            output_type: type[BaseModel] | None = None,
            instructions: str | None = None,
        ) -> ModelResponse:
            raise model_error

    agent = Agent(FailingModel(), toolset=Toolset(tools=()))

    with pytest.raises(AgentError) as exc_info:
        agent.run(
            "Where is my order?",
            context=ToolContext(customer_id="customer-001"),
        )

    assert exc_info.value.code == "model_call_failed"
    assert exc_info.value.__cause__ is model_error

    project_spans = _get_project_spans(capfire)

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 1

    root_span = root_spans[0]
    model_span = model_spans[0]

    root_attributes = root_span["attributes"]
    model_attributes = model_span["attributes"]

    assert root_span["parent"] is None
    _assert_is_direct_child_of(model_span, root_span)

    assert "events" not in model_span
    assert "events" not in root_span

    serialized_project_spans = json.dumps(project_spans, sort_keys=True)
    assert "sensitive provider failure" not in serialized_project_spans

    assert model_attributes["customer_support_agent.model.call.index"] == 1
    assert model_attributes["customer_support_agent.model.call.purpose"] == "agent_loop"
    assert model_attributes["customer_support_agent.model.outcome"] == "exception"
    assert model_attributes["error.type"] == "RuntimeError"
    assert SpanLevel(model_attributes["logfire.level_num"]) >= "error"

    assert root_attributes["customer_support_agent.agent.outcome"] == "error"
    assert (
        root_attributes["customer_support_agent.agent.error.code"]
        == "model_call_failed"
    )
    assert root_attributes["customer_support_agent.agent.model_call.count"] == 1
    assert root_attributes["customer_support_agent.agent.tool_call.count"] == 0
    assert SpanLevel(root_attributes["logfire.level_num"]) >= "error"


def test_agent_trace_records_agent_error_without_changing_error(
    capfire: CaptureLogfire,
) -> None:
    model = ScriptedModel(
        [
            ModelResponse(parts=(TextPart(content="Draft response"),)),
            ModelResponse(),
        ]
    )

    agent = Agent(model, toolset=Toolset(tools=()))

    with pytest.raises(AgentError) as exc_info:
        agent.run(
            "Where is my order?",
            context=ToolContext(customer_id="customer-001"),
        )

    assert exc_info.value.code == "invalid_model_response"
    assert exc_info.value.__cause__ is None

    project_spans = _get_project_spans(capfire)

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 2

    root_span = root_spans[0]
    model_spans.sort(key=lambda span: span["start_time"])

    root_attributes = root_span["attributes"]
    model_attributes = [span["attributes"] for span in model_spans]

    assert root_span["parent"] is None
    assert "events" not in root_span

    for model_span in model_spans:
        _assert_is_direct_child_of(model_span, root_span)
        assert "events" not in model_span

    assert [
        attributes["customer_support_agent.model.call.index"]
        for attributes in model_attributes
    ] == [1, 2]

    assert [
        attributes["customer_support_agent.model.call.purpose"]
        for attributes in model_attributes
    ] == ["agent_loop", "final_result"]

    assert [
        attributes["customer_support_agent.model.outcome"]
        for attributes in model_attributes
    ] == ["success", "success"]

    assert [
        attributes["customer_support_agent.model.response.kind"]
        for attributes in model_attributes
    ] == ["text", "invalid"]

    assert [
        attributes["customer_support_agent.model.tool_call.count"]
        for attributes in model_attributes
    ] == [0, 0]

    assert root_attributes["customer_support_agent.agent.outcome"] == "error"
    assert (
        root_attributes["customer_support_agent.agent.error.code"]
        == "invalid_model_response"
    )
    assert root_attributes["customer_support_agent.agent.model_call.count"] == 2
    assert root_attributes["customer_support_agent.agent.tool_call.count"] == 0
    assert SpanLevel(root_attributes["logfire.level_num"]) >= "error"


def test_agent_trace_records_unexpected_exception_without_sensitive_details(
    capfire: CaptureLogfire,
) -> None:
    unexpected_error = RuntimeError("sensitive unexpected failure")

    class FailingToolset(Toolset):
        def execute(
            self,
            _tool_name: str,
            _arguments: object,
            *,
            context: ToolContext,
        ) -> object:
            raise unexpected_error

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name="unexpected_tool",
                        arguments={"order_id": "secret-order-001"},
                    ),
                )
            )
        ]
    )

    agent = Agent(model, toolset=FailingToolset(tools=()))

    with pytest.raises(RuntimeError) as exc_info:
        agent.run(
            "Use the requested tool.",
            context=ToolContext(customer_id="customer-001"),
        )

    assert exc_info.value is unexpected_error

    project_spans = _get_project_spans(capfire)

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 1

    root_span = root_spans[0]
    model_span = model_spans[0]

    root_attributes = root_span["attributes"]
    model_attributes = model_span["attributes"]

    assert root_span["parent"] is None
    _assert_is_direct_child_of(model_span, root_span)

    assert "events" not in root_span
    assert "events" not in model_span

    serialized_project_spans = json.dumps(project_spans, sort_keys=True)
    assert "sensitive unexpected failure" not in serialized_project_spans

    assert model_attributes["customer_support_agent.model.call.index"] == 1
    assert model_attributes["customer_support_agent.model.call.purpose"] == "agent_loop"
    assert model_attributes["customer_support_agent.model.outcome"] == "success"
    assert (
        model_attributes["customer_support_agent.model.response.kind"] == "tool_calls"
    )
    assert model_attributes["customer_support_agent.model.tool_call.count"] == 1

    assert root_attributes["customer_support_agent.agent.outcome"] == "exception"
    assert root_attributes["error.type"] == "RuntimeError"
    assert "customer_support_agent.agent.error.code" not in root_attributes
    assert root_attributes["customer_support_agent.agent.model_call.count"] == 1
    assert root_attributes["customer_support_agent.agent.tool_call.count"] == 1
    assert SpanLevel(root_attributes["logfire.level_num"]) >= "error"


def test_agent_trace_contains_tool_child_with_safe_attributes(
    capfire: CaptureLogfire,
) -> None:
    @tool
    def lookup_order_status(
        context: ToolContext,
        order_id: str,
        include_history: bool,
    ) -> dict[str, object]:
        """Return the requested order status."""
        return {
            "customer_id": context.customer_id,
            "order_id": order_id,
            "status": "sensitive-tool-result-001",
            "include_history": include_history,
        }

    expected_result = AgentResult(
        message=("Your order is currently processing. sensitive-agent-result-001"),
    )

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=lookup_order_status.definition.name,
                        arguments={
                            "order_id": "secret-order-001",
                            "include_history": True,
                        },
                    ),
                )
            ),
            ModelResponse(
                parts=(
                    TextPart(content="The requested order is currently processing."),
                )
            ),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    agent = Agent(model, toolset=Toolset(tools=(lookup_order_status,)))

    user_message = "sensitive-user-message-001"
    customer_id = "sensitive-customer-001"

    result = agent.run(user_message, context=ToolContext(customer_id=customer_id))

    assert result == expected_result

    project_spans = _get_project_spans(capfire)

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(root_spans) == 1
    assert len(tool_spans) == 1

    root_span = root_spans[0]
    tool_span = tool_spans[0]
    root_attributes = root_span["attributes"]
    tool_attributes = tool_span["attributes"]

    assert root_attributes["customer_support_agent.agent.tool_call.count"] == 1

    _assert_is_direct_child_of(tool_span, root_span)

    assert (
        tool_attributes["customer_support_agent.tool.name"]
        == lookup_order_status.definition.name
    )
    assert tool_attributes["customer_support_agent.tool.argument.names"] == [
        "include_history",
        "order_id",
    ]
    assert tool_attributes["customer_support_agent.tool.outcome"] == "success"
    assert "customer_support_agent.tool.error.code" not in tool_attributes
    assert "error.type" not in tool_attributes

    serialized_project_spans = json.dumps(project_spans, sort_keys=True)

    for sensitive_value in (
        user_message,
        customer_id,
        "secret-order-001",
        "sensitive-tool-result-001",
        "sensitive-agent-result-001",
    ):
        assert sensitive_value not in serialized_project_spans


def test_agent_trace_records_tool_error_without_error_level(
    capfire: CaptureLogfire,
) -> None:
    @tool
    def find_order(order_id: str) -> dict[str, object]:
        """Find an order by ID."""
        return {
            "error": {
                "code": "order_not_found",
                "message": "sensitive-tool-error-message-001",
            }
        }

    expected_result = AgentResult(
        message="I could not find the requested order.",
    )

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=find_order.definition.name,
                        arguments={"order_id": "secret-order-404"},
                    ),
                )
            ),
            ModelResponse(
                parts=(TextPart(content="The requested order was not found."),)
            ),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    agent = Agent(model, toolset=Toolset(tools=(find_order,)))

    result = agent.run(
        "Find my missing order.",
        context=ToolContext(customer_id="sensitive-customer-001"),
    )

    assert result == expected_result

    project_spans = _get_project_spans(capfire)

    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(tool_spans) == 1

    tool_span = tool_spans[0]
    tool_attributes = tool_span["attributes"]

    assert (
        tool_attributes["customer_support_agent.tool.name"]
        == find_order.definition.name
    )
    assert tool_attributes["customer_support_agent.tool.outcome"] == "tool_error"
    assert (
        tool_attributes["customer_support_agent.tool.error.code"] == "order_not_found"
    )
    assert "error.type" not in tool_attributes
    assert "events" not in tool_span

    tool_level_num = tool_attributes.get("logfire.level_num")
    if tool_level_num is not None:
        assert SpanLevel(tool_level_num) < "error"

    serialized_project_spans = json.dumps(project_spans, sort_keys=True)

    for sensitive_value in (
        "secret-order-404",
        "sensitive-tool-error-message-001",
    ):
        assert sensitive_value not in serialized_project_spans


def test_agent_trace_records_tool_exception_without_sensitive_details(
    capfire: CaptureLogfire,
) -> None:
    @tool
    def broken_tool(order_id: str) -> object:
        """Fail while processing an order."""
        raise RuntimeError("sensitive-executor-failure-001")

    expected_result = AgentResult(
        message="I could not process the requested order.",
    )

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=broken_tool.definition.name,
                        arguments={"order_id": "secret-order-500"},
                    ),
                )
            ),
            ModelResponse(parts=(TextPart(content="The requested operation failed."),)),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    agent = Agent(model, toolset=Toolset(tools=(broken_tool,)))

    result = agent.run(
        "Process the requested order.",
        context=ToolContext(customer_id="sensitive-customer-500"),
    )

    assert result == expected_result

    project_spans = _get_project_spans(capfire)

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(root_spans) == 1
    assert len(tool_spans) == 1

    root_attributes = root_spans[0]["attributes"]
    tool_span = tool_spans[0]
    tool_attributes = tool_span["attributes"]

    # Tool 예외가 ToolError로 정규화됐기 때문에 Agent 실행은 완료된다.
    assert root_attributes["customer_support_agent.agent.outcome"] == "success"

    assert tool_attributes["customer_support_agent.tool.outcome"] == "exception"
    assert (
        tool_attributes["customer_support_agent.tool.error.code"]
        == "tool_execution_failed"
    )
    assert tool_attributes["error.type"] == "RuntimeError"
    assert SpanLevel(tool_attributes["logfire.level_num"]) >= "error"

    # 원래 예외를 span 밖으로 전달하지 않았으므로 자동 exception event가 없어야 한다.
    assert "events" not in tool_span

    serialized_project_spans = json.dumps(project_spans, sort_keys=True)

    for forbidden_value in (
        "secret-order-500",
        "sensitive-executor-failure-001",
        "The tool failed unexpectedly; do not assume a result.",
    ):
        assert forbidden_value not in serialized_project_spans


def test_tool_trace_preserves_unknown_tool_error_for_custom_mapping_arguments(
    capfire: CaptureLogfire,
) -> None:
    result = Toolset(tools=()).execute(
        "unknown_tool",
        _HostileMapping(),
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == create_tool_error("unknown_tool")

    project_spans = _get_project_spans(capfire)
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(tool_spans) == 1

    tool_span = tool_spans[0]
    tool_attributes = tool_span["attributes"]

    assert "events" not in tool_span
    assert tool_attributes["customer_support_agent.tool.argument.names"] == []
    assert tool_attributes["customer_support_agent.tool.outcome"] == "tool_error"
    assert tool_attributes["customer_support_agent.tool.error.code"] == "unknown_tool"
    assert _SENSITIVE_MAPPING_ERROR not in json.dumps(project_spans)


def test_tool_trace_returns_custom_mapping_result_without_inspecting_it(
    capfire: CaptureLogfire,
) -> None:
    hostile_result = _HostileMapping()

    @tool
    def return_custom_mapping() -> Mapping[str, object]:
        """Return a custom mapping."""
        return hostile_result

    result = Toolset(tools=(return_custom_mapping,)).execute(
        return_custom_mapping.definition.name,
        {},
        context=ToolContext(customer_id="customer-001"),
    )

    assert result is hostile_result

    project_spans = _get_project_spans(capfire)
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(tool_spans) == 1

    tool_span = tool_spans[0]
    tool_attributes = tool_span["attributes"]

    assert "events" not in tool_span
    assert tool_attributes["customer_support_agent.tool.outcome"] == "success"
    assert _SENSITIVE_MAPPING_ERROR not in json.dumps(project_spans)


def test_tool_trace_ignores_custom_string_argument_keys(
    capfire: CaptureLogfire,
) -> None:
    arguments = {
        "safe_key": "safe value",
        _HostileComparableString("hostile_key"): "sensitive value",
    }

    result = Toolset(tools=()).execute(
        "unknown_tool",
        arguments,
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == create_tool_error("unknown_tool")

    project_spans = _get_project_spans(capfire)
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(tool_spans) == 1

    tool_span = tool_spans[0]
    tool_attributes = tool_span["attributes"]

    assert "events" not in tool_span
    assert tool_attributes["customer_support_agent.tool.argument.names"] == ["safe_key"]
    assert _SENSITIVE_STRING_ERROR not in json.dumps(project_spans)


def test_tool_trace_returns_custom_string_error_code_without_inspecting_it(
    capfire: CaptureLogfire,
) -> None:
    result_with_custom_code: dict[str, object] = {
        "error": {
            "code": _HostileHashString("order_not_found"),
            "message": "No order matched the provided order_id.",
        }
    }

    @tool
    def return_custom_error_code() -> dict[str, object]:
        """Return an error-shaped result with a custom string code."""
        return result_with_custom_code

    result = Toolset(tools=(return_custom_error_code,)).execute(
        return_custom_error_code.definition.name,
        {},
        context=ToolContext(customer_id="customer-001"),
    )

    assert result is result_with_custom_code

    project_spans = _get_project_spans(capfire)
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(tool_spans) == 1

    tool_span = tool_spans[0]
    tool_attributes = tool_span["attributes"]

    assert "events" not in tool_span
    assert tool_attributes["customer_support_agent.tool.outcome"] == "success"
    assert _SENSITIVE_STRING_ERROR not in json.dumps(project_spans)


def test_agent_trace_preserves_order_and_counts_for_repeated_model_and_tool_steps(
    capfire: CaptureLogfire,
) -> None:
    @tool
    def lookup_order(order_id: str) -> dict[str, str]:
        """Look up an order."""
        return {"order_id": order_id}

    @tool
    def lookup_shipment(order_id: str) -> dict[str, str]:
        """Look up a shipment."""
        return {"order_id": order_id}

    expected_result = AgentResult(
        message="The order and shipment were found.",
    )

    model = ScriptedModel(
        [
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-1",
                        name=lookup_order.definition.name,
                        arguments={"order_id": "order-001"},
                    ),
                )
            ),
            ModelResponse(
                parts=(
                    ToolCallPart(
                        id="call-2",
                        name=lookup_shipment.definition.name,
                        arguments={"order_id": "order-001"},
                    ),
                )
            ),
            ModelResponse(
                parts=(TextPart(content="The order and shipment were found."),)
            ),
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    agent = Agent(
        model,
        toolset=Toolset(tools=(lookup_order, lookup_shipment)),
    )

    result = agent.run(
        "Find my order and shipment.",
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == expected_result

    project_spans = _get_project_spans(capfire)

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(root_spans) == 1
    assert len(model_spans) == 4
    assert len(tool_spans) == 2

    root_span = root_spans[0]
    root_attributes = root_span["attributes"]

    assert (
        root_attributes["customer_support_agent.agent.model_call.count"]
        == len(model_spans)
        == 4
    )
    assert (
        root_attributes["customer_support_agent.agent.tool_call.count"]
        == len(tool_spans)
        == 2
    )

    child_spans = [span for span in project_spans if span["name"] != "agent.run"]

    for child_span in child_spans:
        _assert_is_direct_child_of(child_span, root_span)

    child_spans.sort(key=lambda span: span["start_time"])

    assert [span["name"] for span in child_spans] == [
        "model.generate",
        "tool.execute",
        "model.generate",
        "tool.execute",
        "model.generate",
        "model.generate",
    ]

    ordered_tool_spans = [
        span for span in child_spans if span["name"] == "tool.execute"
    ]

    assert [
        span["attributes"]["customer_support_agent.tool.name"]
        for span in ordered_tool_spans
    ] == [
        lookup_order.definition.name,
        lookup_shipment.definition.name,
    ]

    ordered_model_spans = [
        span for span in child_spans if span["name"] == "model.generate"
    ]

    assert [
        span["attributes"]["customer_support_agent.model.call.index"]
        for span in ordered_model_spans
    ] == [1, 2, 3, 4]
