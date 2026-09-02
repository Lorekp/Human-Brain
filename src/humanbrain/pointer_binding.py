"""Position-bound identifier copying over the cell language model."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from io import StringIO
from keyword import iskeyword
from tokenize import NAME, generate_tokens

from humanbrain.language import CellLanguageModel, LearningUpdate, ModelConfig

__all__ = [
    "PointerBoundCellModel",
    "PointerTrainingExample",
    "SymbolInventory",
    "identifier_inventory",
    "materialize",
    "pointerize",
    "python_interface_inventory",
]

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_POINTER = re.compile(r"<@([0-9]+)>")
_POINTER_PREFIX = "<@"
_INVENTORY_SCHEMA = b"humanbrain-symbol-inventory-v1\0marker=<@index>\0"
_MODEL_SCHEMA = (
    b"humanbrain-pointer-bound-cell-model-v2\0bidirectional-prompt-newline-output\0"
)


def _inventory_digest(symbols: tuple[str, ...]) -> str:
    digest = sha256()
    digest.update(_INVENTORY_SCHEMA)
    digest.update(len(symbols).to_bytes(8, "big"))
    for symbol in symbols:
        raw = symbol.encode("ascii")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_symbols(symbols: tuple[str, ...]) -> None:
    if not isinstance(symbols, tuple):
        raise TypeError("symbols must be an ordered tuple")
    if any(
        not isinstance(symbol, str)
        or not symbol.isascii()
        or _IDENTIFIER.fullmatch(symbol) is None
        for symbol in symbols
    ):
        raise ValueError("symbols must be ASCII identifiers")
    if len(symbols) != len(set(symbols)):
        raise ValueError("symbols must be unique")


@dataclass(frozen=True)
class SymbolInventory:
    """Authenticated ordered symbols addressed by stable pointer positions."""

    symbols: tuple[str, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_symbols(self.symbols)
        object.__setattr__(self, "sha256", _inventory_digest(self.symbols))


def identifier_inventory(source: str) -> SymbolInventory:
    """Seal unique non-keyword ASCII Python names in first-occurrence order."""

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    symbols = tuple(
        dict.fromkeys(
            token.string
            for token in generate_tokens(StringIO(source).readline)
            if token.type == NAME
            and token.string.isascii()
            and not iskeyword(token.string)
        )
    )
    return SymbolInventory(symbols)


def python_interface_inventory(source: str, entry_point: str) -> SymbolInventory:
    """Bind one explicitly named top-level callable's ordered interface."""

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if not isinstance(entry_point, str):
        raise TypeError("entry_point must be a string")
    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("source must contain valid Python") from error
    functions = tuple(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
    )
    if len(functions) != 1:
        raise ValueError(
            "source must contain exactly one top-level callable matching entry_point"
        )
    function = functions[0]
    arguments = function.args
    ordered = [function.name]
    ordered.extend(argument.arg for argument in arguments.posonlyargs)
    ordered.extend(argument.arg for argument in arguments.args)
    if arguments.vararg is not None:
        ordered.append(arguments.vararg.arg)
    ordered.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        ordered.append(arguments.kwarg.arg)
    return SymbolInventory(tuple(dict.fromkeys(ordered)))


def _require_inventory(inventory: SymbolInventory) -> None:
    if not isinstance(inventory, SymbolInventory):
        raise TypeError("inventory must be a SymbolInventory")
    _validate_symbols(inventory.symbols)
    if inventory.sha256 != _inventory_digest(inventory.symbols):
        raise ValueError("symbol inventory is not authentic")


def pointerize(text: str, inventory: SymbolInventory) -> str:
    """Replace exact ASCII identifier lexemes with inventory pointers."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _require_inventory(inventory)
    if _POINTER_PREFIX in text:
        raise ValueError("text contains reserved pointer syntax")
    positions = {symbol: index for index, symbol in enumerate(inventory.symbols)}

    def replace(match: re.Match[str]) -> str:
        symbol = match.group(0)
        index = positions.get(symbol)
        return symbol if index is None else f"<@{index}>"

    return _IDENTIFIER.sub(replace, text)


def materialize(text: str, inventory: SymbolInventory) -> str:
    """Resolve every pointer against one authenticated inventory."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _require_inventory(inventory)

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(inventory.symbols):
            raise ValueError(f"pointer index {index} is outside the symbol inventory")
        return inventory.symbols[index]

    result = _POINTER.sub(replace, text)
    if _POINTER_PREFIX in result:
        raise ValueError("malformed pointer marker")
    return result


@dataclass(frozen=True)
class PointerTrainingExample:
    prompt: str
    output: str
    inventory: SymbolInventory

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not isinstance(self.output, str):
            raise TypeError("prompt and output must be strings")
        _require_inventory(self.inventory)


class PointerBoundCellModel:
    """Train and complete identifier-bearing text through positional pointers."""

    def __init__(self, inner: CellLanguageModel) -> None:
        if not isinstance(inner, CellLanguageModel):
            raise TypeError("inner must be a CellLanguageModel")
        self._inner = inner

    @classmethod
    def grow(
        cls,
        examples: Iterable[PointerTrainingExample],
        config: ModelConfig | None = None,
    ) -> PointerBoundCellModel:
        example_tuple = tuple(examples)
        if not example_tuple:
            raise ValueError("at least one pointer training example is required")
        if any(not isinstance(example, PointerTrainingExample) for example in example_tuple):
            raise TypeError("examples must contain PointerTrainingExample values")
        texts = (
            pointerize(example.prompt, example.inventory)
            + "\n"
            + pointerize(example.output, example.inventory)
            for example in example_tuple
        )
        return cls(CellLanguageModel.grow(texts, config))

    @property
    def training_digest(self) -> str:
        digest = sha256()
        digest.update(_MODEL_SCHEMA)
        digest.update(self._inner.training_digest.encode("ascii"))
        return digest.hexdigest()

    @property
    def grown_branches(self) -> int:
        return self._inner.grown_branches

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        """Return the cell model's pointer-bearing completion for audit."""

        if not isinstance(prompt, str) or not isinstance(cue_text, str) or not cue_text:
            raise TypeError("prompt and cue_text must be non-empty strings")
        pointer_prompt = pointerize(prompt, inventory)
        pointer_stop = _pointerize_stop_text(stop_text, inventory)
        return self._inner.complete(
            pointer_prompt + "\n",
            max_new_tokens=max_new_tokens,
            stop_text=pointer_stop,
        )

    def complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        raw = self.raw_complete(
            prompt,
            inventory,
            cue_text=cue_text,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )
        return materialize(raw, inventory)

    def observe_failure(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        feedback_text: str,
    ) -> None:
        """Baseline model has no failure-conditioned working-memory circuit."""

        pointerize(prompt, inventory)
        if not isinstance(cue_text, str) or not cue_text:
            raise ValueError("cue_text must be one non-empty string")
        if not isinstance(feedback_text, str) or not feedback_text:
            raise ValueError("feedback_text must be one non-empty string")

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> LearningUpdate:
        """Learn one pointerized experience without replaying growth examples."""

        if (
            not isinstance(prompt, str)
            or not isinstance(output, str)
            or not isinstance(cue_text, str)
            or not cue_text
        ):
            raise TypeError("prompt, output, and cue_text must be non-empty strings")
        prior_digest = self.training_digest
        experience = (
            pointerize(prompt, inventory)
            + "\n"
            + pointerize(output, inventory)
        )
        update = self._inner.learn(experience, strength=strength)
        return LearningUpdate(
            prior_training_digest=prior_digest,
            training_digest=self.training_digest,
            token_count=update.token_count,
            added_branches=update.added_branches,
            grown_transitions=update.grown_transitions,
        )


def _pointerize_stop_text(
    stop_text: str | tuple[str, ...] | None,
    inventory: SymbolInventory,
) -> str | tuple[str, ...] | None:
    if stop_text is None:
        return None
    if isinstance(stop_text, str):
        return pointerize(stop_text, inventory)
    return tuple(pointerize(marker, inventory) for marker in stop_text)
