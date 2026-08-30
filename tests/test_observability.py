from unittest.mock import Mock

import logfire
import pytest

from customer_support_agent import observability


@pytest.mark.parametrize(
    "send_to_logfire",
    [False, True],
    ids=["local", "hosted"],
)
def test_configure_observability_forwards_explicit_send_policy(
    monkeypatch: pytest.MonkeyPatch,
    send_to_logfire: bool,
) -> None:
    configure = Mock()
    monkeypatch.setattr(logfire, "configure", configure)

    observability.configure_observability(send_to_logfire=send_to_logfire)

    configure.assert_called_once_with(send_to_logfire=send_to_logfire)
