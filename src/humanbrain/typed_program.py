"""Layer-1 typed program values and the deep grow/complete/compile seam.

Layer 0 remains the explicitly named scalar control in :mod:`humanbrain.program`.
This module does not alias or fall back to that control.
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar


def _strict_record(value: object, keys: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _record_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an ordered sequence")
    return value


def _python_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{label} must be a non-keyword Python identifier")
    return value


@dataclass(frozen=True, slots=True)
class ProgramType:
    """One immutable public type in the Layer-1 program language."""

    constructor: str
    arguments: tuple[ProgramType, ...] = ()

    _ARITY: ClassVar[dict[str, int]] = {
        "Int": 0,
        "Bool": 0,
        "Char": 0,
        "String": 0,
        "List": 1,
        "Pair": 2,
    }

    def __post_init__(self) -> None:
        if not isinstance(self.constructor, str) or self.constructor not in self._ARITY:
            raise ValueError("unknown program type constructor")
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(argument, ProgramType) for argument in self.arguments
        ):
            raise TypeError("program type arguments must be a tuple of ProgramType values")
        if len(self.arguments) != self._ARITY[self.constructor]:
            raise ValueError(
                f"{self.constructor} expects {self._ARITY[self.constructor]} type arguments"
            )

    @classmethod
    def Int(cls) -> ProgramType:
        return cls("Int")

    @classmethod
    def Bool(cls) -> ProgramType:
        return cls("Bool")

    @classmethod
    def Char(cls) -> ProgramType:
        return cls("Char")

    @classmethod
    def String(cls) -> ProgramType:
        return cls("String")

    @classmethod
    def List(cls, element: ProgramType) -> ProgramType:
        return cls("List", (element,))

    @classmethod
    def Pair(cls, first: ProgramType, second: ProgramType) -> ProgramType:
        return cls("Pair", (first, second))

    def to_record(self) -> dict[str, object]:
        return {
            "constructor": self.constructor,
            "arguments": [argument.to_record() for argument in self.arguments],
        }

    @classmethod
    def from_record(cls, value: object) -> ProgramType:
        record = _strict_record(
            value,
            frozenset({"constructor", "arguments"}),
            "program type record",
        )
        constructor = record["constructor"]
        if not isinstance(constructor, str):
            raise ValueError("program type constructor must be a string")
        arguments = tuple(
            cls.from_record(argument)
            for argument in _record_sequence(record["arguments"], "program type arguments")
        )
        return cls(constructor=constructor, arguments=arguments)


@dataclass(frozen=True, slots=True)
class ProgramParameter:
    """One ordered, named parameter at the Layer-1 seam."""

    name: str
    type: ProgramType

    def __post_init__(self) -> None:
        _python_identifier(self.name, "program parameter name")
        if not isinstance(self.type, ProgramType):
            raise TypeError("program parameter type must be a ProgramType")

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "type": self.type.to_record()}

    @classmethod
    def from_record(cls, value: object) -> ProgramParameter:
        record = _strict_record(
            value,
            frozenset({"name", "type"}),
            "program parameter record",
        )
        return cls(
            name=_python_identifier(record["name"], "program parameter name"),
            type=ProgramType.from_record(record["type"]),
        )


@dataclass(frozen=True, slots=True)
class TypedSignature:
    """The complete ordered callable type supplied to growth and completion."""

    entry_point: str
    parameters: tuple[ProgramParameter, ...]
    result: ProgramType

    def __post_init__(self) -> None:
        _python_identifier(self.entry_point, "program entry point")
        if not isinstance(self.parameters, tuple) or not all(
            isinstance(parameter, ProgramParameter) for parameter in self.parameters
        ):
            raise TypeError("signature parameters must be a tuple of ProgramParameter values")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("signature parameter names must be unique")
        if not isinstance(self.result, ProgramType):
            raise TypeError("signature result must be a ProgramType")

    def to_record(self) -> dict[str, object]:
        return {
            "entry_point": self.entry_point,
            "parameters": [parameter.to_record() for parameter in self.parameters],
            "result": self.result.to_record(),
        }

    @classmethod
    def from_record(cls, value: object) -> TypedSignature:
        record = _strict_record(
            value,
            frozenset({"entry_point", "parameters", "result"}),
            "typed signature record",
        )
        return cls(
            entry_point=_python_identifier(record["entry_point"], "program entry point"),
            parameters=tuple(
                ProgramParameter.from_record(parameter)
                for parameter in _record_sequence(
                    record["parameters"], "typed signature parameters"
                )
            ),
            result=ProgramType.from_record(record["result"]),
        )


__all__ = ["ProgramParameter", "ProgramType", "TypedSignature"]
