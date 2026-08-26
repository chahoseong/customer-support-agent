import pytest
from pydantic import ValidationError

from customer_support_agent.agent import AgentResult


def test_agent_result_strips_surrounding_whitespace_from_message() -> None:
    result = AgentResult(
        message=" \nOrder status:\n- shipped\n ",
    )

    assert result.message == "Order status:\n- shipped"


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("", id="empty"),
        pytest.param(" \n\t ", id="whitespace-only"),
    ],
)
def test_agent_result_rejects_message_without_displayable_text(
    message: str,
) -> None:
    with pytest.raises(ValidationError):
        AgentResult(message=message)


def test_agent_result_rejects_undefined_fields() -> None:
    with pytest.raises(ValidationError):
        AgentResult.model_validate(
            {
                "message": "The order is being processed.",
                "status": "answered",
            }
        )


def test_agent_result_rejects_message_reassignment() -> None:
    result = AgentResult(
        message="The order is being processed.",
    )
    field_name = "message"

    with pytest.raises(ValidationError):
        setattr(
            result,
            field_name,
            "The order has shipped.",
        )


def test_agent_result_rejects_non_string_message() -> None:
    with pytest.raises(ValidationError):
        AgentResult.model_validate(
            {
                "message": b"The order is being processed.",
            }
        )
