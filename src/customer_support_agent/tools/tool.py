import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import get_type_hints

from pydantic import BaseModel, ValidationError

from .definitions import ToolDefinition as ToolDefinition
from .errors import ToolError, create_tool_error


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

        if self._takes_context:
            return self._executor(context, parsed_arguments)

        return self._executor(parsed_arguments)


def tool[ResultT](executor: Callable[..., ResultT]) -> Tool[ResultT]:
    description = inspect.getdoc(executor)

    if not description:
        raise ValueError("The tool executor requires a non-blank docstring.")

    parameters = tuple(inspect.signature(executor).parameters.values())

    if len(parameters) not in (1, 2):
        raise TypeError(
            "The tool executor must declare one arguments parameter "
            "and may declare one leading ToolContext parameter."
        )
    if any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    ):
        raise TypeError(
            "Tool executor parameters must be positional-or-keyword parameters."
        )
    if any(
        parameter.default is not inspect.Parameter.empty for parameter in parameters
    ):
        raise TypeError("Tool executor parameters must not define default values.")

    type_hints = get_type_hints(executor)

    takes_context = len(parameters) == 2
    if takes_context and type_hints.get(parameters[0].name) is not ToolContext:
        raise TypeError(
            "The first parameter of a two-parameter tool executor must be ToolContext."
        )

    arguments_parameter = parameters[-1]
    if arguments_parameter.name not in type_hints:
        raise TypeError(
            "The tool executor's arguments parameter requires a type annotation."
        )

    arguments_annotation = type_hints[arguments_parameter.name]
    if (
        not isinstance(arguments_annotation, type)
        or arguments_annotation is BaseModel
        or not issubclass(arguments_annotation, BaseModel)
    ):
        raise TypeError(
            "The tool executor requires a concrete BaseModel arguments type."
        )

    arguments_type = arguments_annotation

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
