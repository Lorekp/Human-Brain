"""Versioned checkpoint codec for cell and synapse state."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from humanbrain.cells import AllenGlif3Parameters
from humanbrain.cortex import DendriticCortex, HumanCellParameters
from humanbrain.human_cell_identity import HUMAN_GLIF3_IDENTITY
from humanbrain.language import CellLanguageModel
from humanbrain.tokenizer import BytePhraseTokenizer


FORMAT = "human-cell-brain-checkpoint-v1"


def _pack_phrases(phrases: tuple[bytes, ...]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(phrases) + 1, dtype=np.int64)
    values: list[int] = []
    for index, phrase in enumerate(phrases):
        values.extend(phrase)
        offsets[index + 1] = len(values)
    return offsets, np.asarray(values, dtype=np.uint8)


def _unpack_phrases(offsets: np.ndarray, values: np.ndarray) -> tuple[bytes, ...]:
    return tuple(
        bytes(values[int(offsets[i]) : int(offsets[i + 1])])
        for i in range(len(offsets) - 1)
    )


def save_checkpoint(model: CellLanguageModel, path: Path) -> str:
    if path.suffix != ".npz":
        raise ValueError("cell checkpoint path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    phrase_offsets, phrase_values = _pack_phrases(model._tokenizer.phrases)
    contexts: list[int] = []
    context_offsets = [0]
    targets: list[int] = []
    strengths: list[float] = []
    for context, target, strength in model._cortex.iter_branches():
        contexts.extend(context)
        context_offsets.append(len(contexts))
        targets.append(target)
        strengths.append(strength)
    metadata = {
        "format": FORMAT,
        "training_digest": model.training_digest,
        "cell_count": model._cortex.cell_count,
        "max_order": model._cortex.max_order,
        "grown_transitions": model._cortex.grown_transitions,
        "allen_config_sha256": HUMAN_GLIF3_IDENTITY.config_sha256,
        "parameters": asdict(model._cortex.parameters),
    }
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            phrase_offsets=phrase_offsets,
            phrase_values=phrase_values,
            context_offsets=np.asarray(context_offsets, dtype=np.int64),
            context_values=np.asarray(contexts, dtype=np.int32),
            targets=np.asarray(targets, dtype=np.int32),
            strengths=np.asarray(strengths, dtype=np.float64),
            unigram=model._cortex._unigram,
            threshold_offset=model._cortex._threshold_offset,
        )
    return sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path) -> CellLanguageModel:
    with np.load(path, allow_pickle=False) as state:
        metadata = json.loads(str(state["metadata"]))
        if metadata.get("format") != FORMAT:
            raise ValueError("unsupported cell checkpoint format")
        if metadata.get("allen_config_sha256") != HUMAN_GLIF3_IDENTITY.config_sha256:
            raise ValueError("checkpoint uses a different Allen human cell")
        cell_data = metadata["parameters"].pop("cell")
        parameters = HumanCellParameters(
            cell=AllenGlif3Parameters(**cell_data),
            **metadata["parameters"],
        )
        tokenizer = BytePhraseTokenizer(
            _unpack_phrases(state["phrase_offsets"], state["phrase_values"])
        )
        if tokenizer.vocab_size != int(metadata["cell_count"]):
            raise ValueError("checkpoint vocabulary/cell count mismatch")
        cortex = DendriticCortex(
            cell_count=int(metadata["cell_count"]),
            max_order=int(metadata["max_order"]),
            parameters=parameters,
        )
        offsets = state["context_offsets"]
        values = state["context_values"]
        targets = state["targets"]
        strengths = state["strengths"]
        for index, target in enumerate(targets):
            context = tuple(
                int(value)
                for value in values[int(offsets[index]) : int(offsets[index + 1])]
            )
            slot = cortex._branches.setdefault(context, {})
            slot[int(target)] = float(strengths[index])
        cortex._unigram = np.array(state["unigram"], dtype=np.float64, copy=True)
        cortex._threshold_offset = np.array(
            state["threshold_offset"], dtype=np.float64, copy=True
        )
        cortex._grown_transitions = int(metadata["grown_transitions"])
    return CellLanguageModel(
        tokenizer=tokenizer,
        cortex=cortex,
        training_digest=str(metadata["training_digest"]),
    )
