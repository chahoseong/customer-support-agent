from pathlib import Path

from pydantic_evals import Dataset

from customer_support_agent.agent import AgentResult

from .models import OrderEvalInput, OrderEvalMetadata

_DATASET_PATH = Path(__file__).with_name("scenario_cases.yaml")


def load_order_dataset() -> Dataset[OrderEvalInput, AgentResult, OrderEvalMetadata]:
    return Dataset[
        OrderEvalInput,
        AgentResult,
        OrderEvalMetadata,
    ].from_file(_DATASET_PATH)
