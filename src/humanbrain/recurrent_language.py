"""Language interface for the recurrent adaptive human-cell circuit."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from humanbrain.language import LanguageEvaluation, Prediction
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig, RecurrentRace
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig


@dataclass(frozen=True)
class RecurrentModelConfig:
    max_phrases: int = 0
    min_phrase_frequency: int = 3
    growth_epochs: int = 1
    circuit: RecurrentCircuitConfig = field(default_factory=RecurrentCircuitConfig)


class RecurrentCellLanguageModel:
    """Grow and stream text through one recurrent cell-circuit interface."""

    def __init__(
        self,
        tokenizer: BytePhraseTokenizer,
        circuit: RecurrentCellCircuit,
        training_digest: str,
    ) -> None:
        if tokenizer.vocab_size != circuit.token_cells:
            raise ValueError("token vocabulary and output-cell population must align")
        self._tokenizer = tokenizer
        self._circuit = circuit
        self._training_digest = training_digest

    @classmethod
    def grow(
        cls,
        texts: Iterable[str],
        config: RecurrentModelConfig | None = None,
    ) -> RecurrentCellLanguageModel:
        cfg = config or RecurrentModelConfig()
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
        circuit = RecurrentCellCircuit(
            token_cells=tokenizer.vocab_size,
            config=cfg.circuit,
        )
        for text in corpus:
            circuit.grow_sequence(tokenizer.encode(text), epochs=cfg.growth_epochs)
        circuit.freeze_learning()
        digest = sha256()
        for text in corpus:
            raw = text.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return cls(tokenizer, circuit, training_digest=digest.hexdigest())

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def learned_synapses(self) -> int:
        return self._circuit.learned_synapse_count

    @property
    def reservoir_cells(self) -> int:
        return self._circuit.config.reservoir_cells

    @property
    def active_reservoir_cells(self) -> int:
        return self._circuit.active_reservoir_cells

    @property
    def training_digest(self) -> str:
        return self._training_digest

    def save(self, path: str | Path) -> str:
        """Freeze recurrent cells and synapses and return the artifact SHA-256."""
        from humanbrain.recurrent_artifacts import save_recurrent_checkpoint

        return save_recurrent_checkpoint(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> RecurrentCellLanguageModel:
        from humanbrain.recurrent_artifacts import load_recurrent_checkpoint

        return load_recurrent_checkpoint(Path(path))

    def _prediction(self, race: RecurrentRace) -> Prediction:
        return Prediction(
            token_id=race.winner,
            text=self._tokenizer.decode([race.winner]),
            spike_time_ms=race.spike_time_ms,
            peak_voltage_mv=race.peak_voltage_mv,
            active_cells=race.active_cells,
        )

    def predict_next(self, prompt: str) -> Prediction:
        ids = self._tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prediction prompt cannot be empty")
        return self._prediction(self._circuit.predict_next(ids))

    def complete(self, prompt: str, max_new_tokens: int = 32) -> str:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        ids = self._tokenizer.encode(prompt)
        if not ids:
            raise ValueError("completion prompt cannot be empty")
        self._circuit.reset_sequence()
        for token_id in ids:
            self._circuit.observe(token_id)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            race = self._circuit.race()
            generated.append(race.winner)
            self._circuit.observe(race.winner)
        return self._tokenizer.decode(generated)

    def evaluate(self, text: str, max_predictions: int | None = None) -> LanguageEvaluation:
        """Teacher-force held-out text while preserving recurrent cell state."""
        ids = self._tokenizer.encode(text)
        available = max(0, len(ids) - 1)
        count = available if max_predictions is None else min(available, max_predictions)
        if count < 1:
            raise ValueError("evaluation text must contain at least two tokens")
        self._circuit.reset_sequence()
        self._circuit.observe(ids[0])
        correct = 0
        spike_time_sum = 0.0
        started = perf_counter()
        for target_index in range(1, count + 1):
            race = self._circuit.race()
            correct += int(race.winner == ids[target_index])
            spike_time_sum += race.spike_time_ms
            self._circuit.observe(ids[target_index])
        elapsed = perf_counter() - started
        return LanguageEvaluation(
            predictions=count,
            correct=correct,
            accuracy=correct / count,
            mean_spike_time_ms=spike_time_sum / count,
            wall_seconds=elapsed,
            tokens_per_second=count / elapsed if elapsed else float("inf"),
        )
