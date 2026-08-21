"""Deep interface for autoregressive prediction by simulated cells."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Iterable

from humanbrain.cortex import CellRace, DendriticCortex, HumanCellParameters
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig


@dataclass(frozen=True)
class ModelConfig:
    max_order: int = 8
    max_phrases: int = 512
    min_phrase_frequency: int = 2
    random_seed: int = 7
    cell_parameters: HumanCellParameters = HumanCellParameters()


@dataclass(frozen=True)
class Prediction:
    token_id: int
    text: str
    spike_time_ms: float
    peak_voltage_mv: float
    active_cells: int


@dataclass(frozen=True)
class LanguageEvaluation:
    predictions: int
    correct: int
    accuracy: float
    mean_spike_time_ms: float
    wall_seconds: float
    tokens_per_second: float


class CellLanguageModel:
    """Grow, predict, and complete text through one small interface."""

    def __init__(
        self,
        tokenizer: BytePhraseTokenizer,
        cortex: DendriticCortex,
        training_digest: str,
    ) -> None:
        if tokenizer.vocab_size != cortex.cell_count:
            raise ValueError("token vocabulary and cortical sheet must align")
        self._tokenizer = tokenizer
        self._cortex = cortex
        self._training_digest = training_digest

    @classmethod
    def grow(
        cls,
        texts: Iterable[str],
        config: ModelConfig | None = None,
    ) -> "CellLanguageModel":
        cfg = config or ModelConfig()
        corpus = tuple(str(text) for text in texts)
        if not corpus or not any(corpus):
            raise ValueError("at least one non-empty growth text is required")
        tokenizer = BytePhraseTokenizer.fit(
            corpus,
            TokenizerConfig(
                max_phrases=cfg.max_phrases,
                min_frequency=cfg.min_phrase_frequency,
            ),
        )
        cortex = DendriticCortex(
            cell_count=tokenizer.vocab_size,
            max_order=cfg.max_order,
            parameters=cfg.cell_parameters,
            seed=cfg.random_seed,
        )
        for text in corpus:
            cortex.grow_sequence(tokenizer.encode(text))
        digest = sha256()
        for text in corpus:
            raw = text.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return cls(tokenizer, cortex, training_digest=digest.hexdigest())

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def grown_branches(self) -> int:
        return self._cortex.branch_count

    @property
    def training_digest(self) -> str:
        return self._training_digest

    def save(self, path: str | Path) -> str:
        """Freeze learned cell/synapse state and return its SHA-256."""
        from humanbrain.artifacts import save_checkpoint

        return save_checkpoint(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "CellLanguageModel":
        from humanbrain.artifacts import load_checkpoint

        return load_checkpoint(Path(path))

    def predict_next(self, prompt: str) -> Prediction:
        ids = self._tokenizer.encode(prompt)
        race = self._cortex.race(ids)
        return self._prediction(race)

    def _prediction(self, race: CellRace) -> Prediction:
        return Prediction(
            token_id=race.winner,
            text=self._tokenizer.decode([race.winner]),
            spike_time_ms=race.spike_time_ms,
            peak_voltage_mv=race.peak_voltage_mv,
            active_cells=race.active_cells,
        )

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        stop_text: str | None = None,
    ) -> str:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        ids = self._tokenizer.encode(prompt)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            race = self._cortex.race(ids)
            ids.append(race.winner)
            generated.append(race.winner)
            if stop_text is not None:
                decoded = self._tokenizer.decode(generated)
                stop_index = decoded.find(stop_text)
                if stop_index >= 0:
                    return decoded[: stop_index + len(stop_text)]
        return self._tokenizer.decode(generated)

    def evaluate(self, text: str, max_predictions: int | None = None) -> LanguageEvaluation:
        """Teacher-forced held-out next-token accuracy through cell races."""
        ids = self._tokenizer.encode(text)
        available = max(0, len(ids) - 1)
        count = available if max_predictions is None else min(available, max_predictions)
        if count < 1:
            raise ValueError("evaluation text must contain at least two tokens")
        correct = 0
        spike_time_sum = 0.0
        started = perf_counter()
        for target_index in range(1, count + 1):
            race = self._cortex.race(ids[:target_index])
            correct += int(race.winner == ids[target_index])
            spike_time_sum += race.spike_time_ms
        elapsed = perf_counter() - started
        return LanguageEvaluation(
            predictions=count,
            correct=correct,
            accuracy=correct / count,
            mean_spike_time_ms=spike_time_sum / count,
            wall_seconds=elapsed,
            tokens_per_second=count / elapsed if elapsed else float("inf"),
        )

    def measure_completion(self, prompt: str, max_new_tokens: int = 32) -> dict[str, float | str]:
        started = perf_counter()
        text = self.complete(prompt, max_new_tokens=max_new_tokens)
        elapsed = perf_counter() - started
        return {
            "text": text,
            "seconds": elapsed,
            "tokens_per_second": max_new_tokens / elapsed if elapsed else float("inf"),
        }
