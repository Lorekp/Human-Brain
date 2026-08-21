"""Frozen language-artifact format for recurrent human-cell circuits."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from humanbrain.artifacts import _pack_phrases, _unpack_phrases
from humanbrain.recurrent import RecurrentCellCircuit
from humanbrain.recurrent_language import RecurrentCellLanguageModel
from humanbrain.tokenizer import BytePhraseTokenizer

FORMAT = "recurrent-human-cell-checkpoint-v3"


def save_recurrent_checkpoint(model: RecurrentCellLanguageModel, path: Path) -> str:
    if path.suffix != ".npz":
        raise ValueError("recurrent checkpoint path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit = model._circuit
    cell_synapse_state = circuit.capture_cell_synapse_state()
    phrase_offsets, phrase_values = _pack_phrases(model._tokenizer.phrases)
    metadata = {
        "format": FORMAT,
        "training_digest": model.training_digest,
        **cell_synapse_state.metadata,
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "phrase_offsets": phrase_offsets,
        "phrase_values": phrase_values,
        **cell_synapse_state.arrays,
    }
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return sha256(path.read_bytes()).hexdigest()


def load_recurrent_checkpoint(path: Path) -> RecurrentCellLanguageModel:
    with np.load(path, allow_pickle=False) as state:
        metadata = json.loads(str(state["metadata"]))
        if metadata.get("format") != FORMAT:
            raise ValueError("unsupported recurrent checkpoint format")
        tokenizer = BytePhraseTokenizer(
            _unpack_phrases(state["phrase_offsets"], state["phrase_values"])
        )
        token_cells = int(metadata["token_cells"])
        if tokenizer.vocab_size != token_cells:
            raise ValueError("checkpoint vocabulary/output-cell count mismatch")
        circuit_state = RecurrentCellCircuit.decode_cell_synapse_state(metadata, state)
        circuit = RecurrentCellCircuit.restore_cell_synapse_state(circuit_state)
    return RecurrentCellLanguageModel(
        tokenizer=tokenizer,
        circuit=circuit,
        training_digest=str(metadata["training_digest"]),
    )
