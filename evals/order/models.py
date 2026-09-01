from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

type InformationSource = Literal[
    "customer_orders",
    "order",
    "shipment",
    "cancellation_policy",
]

type InformationSourceOutcome = Literal[
    "available",
    "unavailable",
    "execution_failed",
]

type ScenarioId = Annotated[
    str,
    Field(strict=True, pattern=r"^scenario-[1-9][0-9]*$"),
]

type ResponseCriterion = Annotated[
    str,
    Field(strict=True, min_length=1),
]

type ExecutionCondition = Literal[
    "default",
    "shipment_information_failure",
]

type OrderId = Annotated[str, Field(strict=True, min_length=1)]


class OrderEvalInput(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    user_message: Annotated[str, Field(strict=True, min_length=1)]
    customer_id: Annotated[str, Field(strict=True, min_length=1)]
    execution_condition: ExecutionCondition


class InformationSourceExpectation(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    source: InformationSource
    order_id: OrderId | None = None
    outcome: InformationSourceOutcome

    @model_validator(mode="after")
    def validate_order_id_matches_source(self) -> Self:
        requires_order_id = self.source in {"order", "shipment"}
        has_order_id = self.order_id is not None

        if requires_order_id != has_order_id:
            raise ValueError(
                "order_id is required only for order and shipment information sources"
            )

        return self


class OrderEvalMetadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    scenario_id: ScenarioId
    required_information_sources: frozenset[InformationSourceExpectation]
    forbidden_information_sources: frozenset[InformationSource]
    required_response_criteria: tuple[ResponseCriterion, ...]
    forbidden_response_criteria: tuple[ResponseCriterion, ...]

    @model_validator(mode="after")
    def validate_information_sources_do_not_overlap(self) -> Self:
        required_source_types = {
            expectation.source for expectation in self.required_information_sources
        }

        if not required_source_types.isdisjoint(self.forbidden_information_sources):
            raise ValueError(
                "required and forbidden information sources must not overlap"
            )

        return self
