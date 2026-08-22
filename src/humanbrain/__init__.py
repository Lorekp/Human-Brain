"""Human-cell language prediction.

Tokenization is ordinary software plumbing. Learned state and next-token
selection live in the simulated cortical cells exposed by CellLanguageModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from humanbrain.language import (
        CellLanguageModel,
        LanguageEvaluation,
        ModelConfig,
        Prediction,
    )

__all__ = ["CellLanguageModel", "LanguageEvaluation", "ModelConfig", "Prediction"]
_LANGUAGE_EXPORTS = frozenset(__all__)
__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    if name not in _LANGUAGE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from humanbrain import language

    value = getattr(language, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LANGUAGE_EXPORTS)
