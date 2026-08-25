import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from .errors import ToolError, create_tool_error


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolContext:
    customer_id: str


class Tool[ResultT]:
    def __init__(
        self,
        *,
        definition: ToolDefinition,
        arguments_type: type[BaseModel],
        executor: Callable[..., ResultT],
        takes_context: bool,
    ) -> None:
        self.definition = definition
        self._arguments_type = arguments_type
        self._executor = executor
        self._takes_context = takes_context

    def __call__(
        self,
        arguments: object,
        *,
        context: ToolContext | None = None,
    ) -> ResultT | ToolError:
        if self._takes_context and context is None:
            raise TypeError("The tool executor requires ToolContext.")

        try:
            parsed_arguments = self._arguments_type.model_validate(arguments)
        except ValidationError:
            return create_tool_error("invalid_arguments")

        validated_arguments = {
            name: getattr(parsed_arguments, name)
            for name in self._arguments_type.model_fields
        }

        if self._takes_context:
            return self._executor(context, **validated_arguments)

        return self._executor(**validated_arguments)


def tool[ResultT](executor: Callable[..., ResultT]) -> Tool[ResultT]:
    description = inspect.getdoc(executor)

    if not description:
        raise ValueError("The tool executor requires a non-blank docstring.")

    parameters = tuple(inspect.signature(executor).parameters.values())

    if any(
        parameter.kind
        not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for parameter in parameters
    ):
        raise TypeError(
            "Tool executor parameters must be positional-or-keyword "
            "or keyword-only parameters."
        )
    type_hints = get_type_hints(executor, include_extras=True)

    context_parameters = tuple(
        parameter
        for parameter in parameters
        if type_hints.get(parameter.name) is ToolContext
    )
    if context_parameters and (
        len(context_parameters) != 1
        or context_parameters[0].name != parameters[0].name
        or context_parameters[0].kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
    ):
        raise TypeError(
            "ToolContext must be the first positional-or-keyword parameter."
        )

    takes_context = bool(context_parameters)
    if takes_context and parameters[0].default is not inspect.Parameter.empty:
        raise TypeError("The ToolContext parameter must not define a default value.")

    exposed_parameters = parameters[1:] if takes_context else parameters

    for parameter in exposed_parameters:
        if parameter.name not in type_hints:
            raise TypeError("Tool executor parameters require type annotations.")

    field_definitions: dict[str, Any] = {
        parameter.name: (
            type_hints[parameter.name],
            ... if parameter.default is inspect.Parameter.empty else parameter.default,
        )
        for parameter in exposed_parameters
    }
    arguments_type = create_model(
        f"{executor.__name__}Arguments",
        __config__=ConfigDict(strict=True, extra="forbid"),
        **field_definitions,
    )

    return Tool(
        definition=ToolDefinition(
            name=executor.__name__,
            description=description,
            parameters=arguments_type.model_json_schema(),
        ),
        arguments_type=arguments_type,
        executor=executor,
        takes_context=takes_context,
    )
