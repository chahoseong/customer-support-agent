import logfire


def configure_observability(*, send_to_logfire: bool) -> None:
    logfire.configure(send_to_logfire=send_to_logfire)
