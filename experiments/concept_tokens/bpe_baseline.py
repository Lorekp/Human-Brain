"""Backward-compat shim: BpeTokenizer moved to :mod:`humanbrain.bpe_tokenizer`.

The concept-token study's BPE baseline was promoted into the package so
benchmark harnesses (``benchmarks/qa_experiments --tokenizer bpe4096``) can
import it alongside BytePhraseTokenizer/ConceptTokenizer. This module keeps
the historical ``from bpe_baseline import BpeTokenizer`` import path working
for ``experiments/concept_tokens/run_study.py``.
"""

from humanbrain.bpe_tokenizer import BpeTokenizer  # noqa: F401

__all__ = ["BpeTokenizer"]
