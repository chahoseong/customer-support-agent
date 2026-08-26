from customer_support_agent.agent import Agent
from customer_support_agent.models import ChatModel
from customer_support_agent.tools import Toolset
from customer_support_agent.tools.order import find_order, get_customer_orders
from customer_support_agent.tools.policy import get_cancellation_policy
from customer_support_agent.tools.shipment import find_shipment

_CUSTOMER_SUPPORT_INSTRUCTIONS = """You are a customer support agent for order-related questions.

Use the available tools to retrieve the information needed to answer the
customer. Base factual claims about orders, shipments, and cancellation
eligibility only on information returned by the tools. Do not invent or
assume missing facts.

When the customer's request does not identify the relevant order, retrieve
their available orders and ask them to clarify which order they mean.

When answering whether an order can be cancelled, retrieve both the order
status and the cancellation policy, then answer using only those facts.

If a tool cannot provide the requested information, explain what information
could not be confirmed and guide the customer on what they can provide or try
next. Do not expose internal tool names, error codes, or implementation
details.

Return a concise, customer-facing response in the required structured output."""

CUSTOMER_SUPPORT_TOOLSET = Toolset(
    tools=(
        get_customer_orders,
        find_order,
        find_shipment,
        get_cancellation_policy,
    )
)


def create_customer_support_agent(model: ChatModel) -> Agent:
    return Agent(
        model,
        CUSTOMER_SUPPORT_TOOLSET,
        instructions=_CUSTOMER_SUPPORT_INSTRUCTIONS,
    )
