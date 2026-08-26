import pytest

import customer_support_agent.tools.policy as policy_module
from customer_support_agent.domain.models import CancellationPolicy
from customer_support_agent.tools.policy import get_cancellation_policy


def test_get_cancellation_policy_returns_canonical_cancellable_statuses() -> None:
    result = get_cancellation_policy({})

    assert result == {"cancellable_statuses": ["processing"]}


def test_get_cancellation_policy_returns_statuses_in_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CancellationPolicy(
        cancellable_statuses=frozenset({"shipped", "processing"})
    )
    monkeypatch.setattr(
        policy_module,
        "CANCELLATION_POLICY",
        policy,
    )

    result = get_cancellation_policy({})

    assert result == {"cancellable_statuses": ["processing", "shipped"]}


def test_get_cancellation_policy_definition_describes_global_policy_lookup() -> None:
    definition = get_cancellation_policy.definition

    assert definition.name == "get_cancellation_policy"
    assert definition.description == (
        "Retrieve the global cancellation policy as the order statuses "
        "that permit cancellation."
    )
    assert definition.parameters["type"] == "object"
    assert definition.parameters["properties"] == {}
    assert definition.parameters["additionalProperties"] is False
