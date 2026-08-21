"""Human-cell language prediction.

Tokenization is ordinary software plumbing. Learned state and next-token
selection live in the simulated cortical cells exposed by CellLanguageModel.
"""

from humanbrain.language import CellLanguageModel, LanguageEvaluation, ModelConfig, Prediction

__all__ = ["CellLanguageModel", "LanguageEvaluation", "ModelConfig", "Prediction"]
__version__ = "0.1.0"
