import json
from collections.abc import Sequence

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

from .scripted_model import ScriptedModel


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

    exported_spans = capfire.exporter.exported_spans_as_dict()

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate"}
    ]

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]

    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 2

    root_span = root_spans[0]
    model_spans.sort(key=lambda span: span["start_time"])

    assert root_span["parent"] is None

    root_context = root_span["context"]
    root_attributes = root_span["attributes"]

    assert root_attributes["customer_support_agent.agent.outcome"] == "success"
    assert root_attributes["customer_support_agent.agent.model_call.count"] == 2
    assert root_attributes["customer_support_agent.agent.tool_call.count"] == 0

    for model_span in model_spans:
        assert model_span["context"]["trace_id"] == root_context["trace_id"]
        assert model_span["parent"] is not None
        assert model_span["parent"]["span_id"] == root_context["span_id"]

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

    exported_spans = capfire.exporter.exported_spans_as_dict()

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate"}
    ]
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
            assert model_span["parent"] is not None
            assert model_span["parent"]["span_id"] == root_context["span_id"]


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

    exported_spans = capfire.exporter.exported_spans_as_dict()

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate"}
    ]

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 1

    root_span = root_spans[0]
    model_span = model_spans[0]

    root_context = root_span["context"]
    root_attributes = root_span["attributes"]
    model_attributes = model_span["attributes"]

    assert root_span["parent"] is None
    assert model_span["context"]["trace_id"] == root_context["trace_id"]
    assert model_span["parent"] is not None
    assert model_span["parent"]["span_id"] == root_context["span_id"]

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

    exported_spans = capfire.exporter.exported_spans_as_dict()

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate"}
    ]

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 2

    root_span = root_spans[0]
    model_spans.sort(key=lambda span: span["start_time"])

    root_context = root_span["context"]
    root_attributes = root_span["attributes"]
    model_attributes = [span["attributes"] for span in model_spans]

    assert root_span["parent"] is None
    assert "events" not in root_span

    for model_span in model_spans:
        assert model_span["context"]["trace_id"] == root_context["trace_id"]
        assert model_span["parent"] is not None
        assert model_span["parent"]["span_id"] == root_context["span_id"]
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

    exported_spans = capfire.exporter.exported_spans_as_dict()

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate"}
    ]

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    model_spans = [span for span in project_spans if span["name"] == "model.generate"]

    assert len(root_spans) == 1
    assert len(model_spans) == 1

    root_span = root_spans[0]
    model_span = model_spans[0]

    root_context = root_span["context"]
    root_attributes = root_span["attributes"]
    model_attributes = model_span["attributes"]

    assert root_span["parent"] is None
    assert model_span["context"]["trace_id"] == root_context["trace_id"]
    assert model_span["parent"] is not None
    assert model_span["parent"]["span_id"] == root_context["span_id"]

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

    exported_spans = capfire.exporter.exported_spans_as_dict(
        parse_json_attributes=True,
    )

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate", "tool.execute"}
    ]

    root_spans = [span for span in project_spans if span["name"] == "agent.run"]
    tool_spans = [span for span in project_spans if span["name"] == "tool.execute"]

    assert len(root_spans) == 1
    assert len(tool_spans) == 1

    root_span = root_spans[0]
    tool_span = tool_spans[0]
    root_attributes = root_span["attributes"]
    tool_attributes = tool_span["attributes"]

    assert root_attributes["customer_support_agent.agent.tool_call.count"] == 1

    assert tool_span["context"]["trace_id"] == root_span["context"]["trace_id"]
    assert tool_span["parent"] is not None
    assert tool_span["parent"]["span_id"] == root_span["context"]["span_id"]

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

    exported_spans = capfire.exporter.exported_spans_as_dict(
        parse_json_attributes=True,
    )

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate", "tool.execute"}
    ]

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

    exported_spans = capfire.exporter.exported_spans_as_dict(
        parse_json_attributes=True,
    )

    project_spans = [
        span
        for span in exported_spans
        if span["name"] in {"agent.run", "model.generate", "tool.execute"}
    ]

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
