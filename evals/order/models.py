from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from customer_support_agent.tools.errors import ToolErrorCode

type ExpectedToolOutcome = Literal["success"] | ToolErrorCode

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


class ExpectedToolUse(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tool_name: str
    expected_arguments: dict[str, object]
    expected_outcome: ExpectedToolOutcome


class OrderEvalInput(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    user_message: Annotated[str, Field(strict=True, min_length=1)]
    customer_id: Annotated[str, Field(strict=True, min_length=1)]
    execution_condition: ExecutionCondition


class OrderEvalMetadata(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    scenario_id: ScenarioId
    required_tool_uses: tuple[ExpectedToolUse, ...]
    forbidden_tools: frozenset[str]
    required_tool_sequence: tuple[str, ...]
    required_response_criteria: tuple[ResponseCriterion, ...]
    forbidden_response_criteria: tuple[ResponseCriterion, ...]

    @model_validator(mode="after")
    def validate_required_tool_names_are_unique(self) -> Self:
        required_tool_names = [
            tool_use.tool_name for tool_use in self.required_tool_uses
        ]

        if len(required_tool_names) != len(set(required_tool_names)):
            raise ValueError("required tool names must be unique")

        return self

    @model_validator(mode="after")
    def validate_required_and_forbidden_tools_do_not_overlap(self) -> Self:
        required_tool_names = {
            tool_use.tool_name for tool_use in self.required_tool_uses
        }

        if not required_tool_names.isdisjoint(self.forbidden_tools):
            raise ValueError("required and forbidden tools must not overlap")

        return self

    @model_validator(mode="after")
    def validate_required_tool_sequence_references_only_required_tools(self) -> Self:
        required_tool_names = {
            tool_use.tool_name for tool_use in self.required_tool_uses
        }

        if not set(self.required_tool_sequence).issubset(required_tool_names):
            raise ValueError(
                "required tool sequence must reference only required tools"
            )

        return self

    @model_validator(mode="after")
    def validate_required_tool_sequence_names_are_unique(self) -> Self:
        if len(self.required_tool_sequence) != len(set(self.required_tool_sequence)):
            raise ValueError(
                "required tool sequence must not contain duplicate tool names"
            )

        return self
