from customer_support_agent.agent import AgentResult
from customer_support_agent.customer_support import (
    CUSTOMER_SUPPORT_TOOLSET,
    create_customer_support_agent,
)
from customer_support_agent.messages import (
    ModelResponse,
    StructuredOutputPart,
)
from customer_support_agent.tools import ToolContext
from customer_support_agent.tools.order import (
    find_order,
    get_customer_orders,
)
from customer_support_agent.tools.policy import get_cancellation_policy
from customer_support_agent.tools.shipment import find_shipment

from .scripted_model import ScriptedModel

EXPECTED_CUSTOMER_SUPPORT_INSTRUCTIONS = """You are a customer support agent for order-related questions.

Support only questions about order status, shipment status, and cancellation
eligibility. Provide information only. Do not claim that you changed or
cancelled an order.

Use the available tools to retrieve the information needed to answer the
customer. Base factual claims about orders, shipments, and cancellation
eligibility only on information returned by the tools. Use structured tool
errors when deciding what can be confirmed. Do not invent or assume missing
facts.

When the customer's request does not identify the relevant order, retrieve
their available orders and ask them to clarify which order they mean.

For shipment status questions, retrieve the order first. Retrieve shipment
information only when the order status is shipped. If the order cannot be
retrieved, do not retrieve shipment information. Do not assume a shipment
status that was not returned by a tool.

For cancellation eligibility questions, retrieve the order first. If the
order cannot be retrieved, do not retrieve the cancellation policy. Otherwise,
retrieve the cancellation policy. An order is cancellable only when its status
is included in cancellable_statuses. State only whether the order is
cancellable, and do not invent cancellation reasons that are not present in
the policy.

If a tool cannot provide the requested information, explain what information
could not be confirmed and guide the customer on what they can provide or try
next. Do not expose tool names, tool call IDs, internal error codes, exception
details, or model and provider details.

Return a concise, customer-facing response in the required structured output."""


def test_customer_support_toolset_exposes_configured_tool_definitions() -> None:
    assert CUSTOMER_SUPPORT_TOOLSET.definitions == (
        get_customer_orders.definition,
        find_order.definition,
        find_shipment.definition,
        get_cancellation_policy.definition,
    )


def test_customer_support_agent_passes_configured_tools_and_instructions_to_model() -> (
    None
):
    expected_result = AgentResult(message="Expected response")

    model = ScriptedModel(
        [
            ModelResponse(parts=(StructuredOutputPart(output=expected_result),)),
        ]
    )

    agent = create_customer_support_agent(model)

    result = agent.run(
        "Where is my order?",
        context=ToolContext(customer_id="customer-001"),
    )

    assert result == expected_result
    assert model.received_tools == [
        CUSTOMER_SUPPORT_TOOLSET.definitions,
    ]
    assert model.received_instructions == [
        EXPECTED_CUSTOMER_SUPPORT_INSTRUCTIONS,
    ]
