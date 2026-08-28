"""Run the Customer Support Agent in a local Web Chat."""

import os

from customer_support_agent.customer_support import create_customer_support_agent
from customer_support_agent.models import OpenAIChatModel
from customer_support_agent.web_chat import create_chat_interface


def _get_required_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"{name} environment variable is required")

    return value.strip()


def main() -> None:
    model = OpenAIChatModel(
        base_url=_get_required_env("LLM_BASE_URL"),
        model_name=_get_required_env("LLM_MODEL_NAME"),
        api_key=os.getenv("LLM_API_KEY", "").strip() or "no-api-key",
    )
    agent = create_customer_support_agent(model)
    interface = create_chat_interface(agent)

    interface.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
