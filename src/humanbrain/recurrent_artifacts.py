"""Frozen artifact format for recurrent human-cell language circuits."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from humanbrain.artifacts import _pack_phrases, _unpack_phrases
from humanbrain.cells import ALLEN_GLIF5_CONFIG_SHA256, AllenGlif5Parameters
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel
from humanbrain.tokenizer import BytePhraseTokenizer

FORMAT = "recurrent-human-cell-checkpoint-v3"

_DEVELOPMENTAL_ARRAYS = {
    "_development_cell_order": ("development_cell_order", np.int32),
    "_development_active_mask": ("development_active_mask", np.bool_),
    "_structural_scores": ("structural_scores", np.float64),
    "_structural_exposure": ("structural_exposure", np.float64),
    "_homeostatic_gain": ("homeostatic_gain", np.float64),
}


def save_recurrent_checkpoint(model: RecurrentCellLanguageModel, path: Path) -> str:
    if path.suffix != ".npz":
        raise ValueError("recurrent checkpoint path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit = model._circuit
    if not circuit.learning_frozen:
        raise RuntimeError("freeze recurrent readout learning before checkpointing")
    if circuit.config.readout_learning_rule == "ridge":
        circuit._flush_growth_statistics()
        if circuit._normal_matrix is None or circuit._target_moments is None:
            raise RuntimeError("cannot checkpoint a recurrent circuit before growth")
    elif circuit._normal_matrix is not None or circuit._target_moments is not None:
        raise RuntimeError("local readout must not contain ridge statistics")
    phrase_offsets, phrase_values = _pack_phrases(model._tokenizer.phrases)
    developmental_arrays_present = [
        hasattr(circuit, name) for name in _DEVELOPMENTAL_ARRAYS
    ]
    if any(developmental_arrays_present) and not all(developmental_arrays_present):
        raise RuntimeError("developmental checkpoint state is incomplete")
    developmental_state = all(developmental_arrays_present)
    metadata = {
        "format": FORMAT,
        "training_digest": model.training_digest,
        "token_cells": circuit.token_cells,
        "allen_glif5_config_sha256": ALLEN_GLIF5_CONFIG_SHA256,
        "cell_parameters": asdict(circuit.cell_parameters),
        "circuit_config": asdict(circuit.config),
        "growth_examples": circuit._growth_examples,
        "local_updates": circuit._local_updates,
        "runtime_active_reservoir_cells": circuit.active_reservoir_cells,
        "developmental_state": developmental_state,
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "phrase_offsets": phrase_offsets,
        "phrase_values": phrase_values,
        "afferent_targets": circuit._afferent_targets,
        "recurrent_targets": circuit._recurrent_targets,
        "recurrent_weights_pa": circuit._recurrent_weights_pa,
        "output_heterogeneity_pa": circuit._output_heterogeneity_pa,
        "readout_synapses_pa": circuit._readout_synapses_pa,
        "pacemaker_synapses_pa": circuit._pacemaker_synapses_pa,
    }
    if circuit.config.readout_learning_rule == "ridge":
        arrays["normal_matrix"] = circuit._normal_matrix
        arrays["target_moments"] = circuit._target_moments
    if developmental_state:
        for attribute, (key, _) in _DEVELOPMENTAL_ARRAYS.items():
            arrays[key] = getattr(circuit, attribute)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return sha256(path.read_bytes()).hexdigest()


def load_recurrent_checkpoint(path: Path) -> RecurrentCellLanguageModel:
    with np.load(path, allow_pickle=False) as state:
        metadata = json.loads(str(state["metadata"]))
        if metadata.get("format") != FORMAT:
            raise ValueError("unsupported recurrent checkpoint format")
        if metadata.get("allen_glif5_config_sha256") != ALLEN_GLIF5_CONFIG_SHA256:
            raise ValueError("checkpoint uses a different adaptive Allen human cell")
        tokenizer = BytePhraseTokenizer(
            _unpack_phrases(state["phrase_offsets"], state["phrase_values"])
        )
        token_cells = int(metadata["token_cells"])
        if tokenizer.vocab_size != token_cells:
            raise ValueError("checkpoint vocabulary/output-cell count mismatch")
        circuit = RecurrentCellCircuit(
            token_cells=token_cells,
            config=RecurrentCircuitConfig(**metadata["circuit_config"]),
            cell_parameters=AllenGlif5Parameters(**metadata["cell_parameters"]),
        )
        arrays = {
            "_afferent_targets": ("afferent_targets", np.int32),
            "_recurrent_targets": ("recurrent_targets", np.int32),
            "_recurrent_weights_pa": ("recurrent_weights_pa", np.float64),
            "_output_heterogeneity_pa": ("output_heterogeneity_pa", np.float64),
            "_readout_synapses_pa": ("readout_synapses_pa", np.float64),
            "_pacemaker_synapses_pa": ("pacemaker_synapses_pa", np.float64),
        }
        if circuit.config.readout_learning_rule == "ridge":
            arrays["_normal_matrix"] = ("normal_matrix", np.float64)
            arrays["_target_moments"] = ("target_moments", np.float64)
        for attribute, (key, dtype) in arrays.items():
            existing = getattr(circuit, attribute)
            if existing is None and attribute == "_normal_matrix":
                expected = (
                    circuit._growth_feature_count,
                    circuit._growth_feature_count,
                )
            elif existing is None and attribute == "_target_moments":
                expected = (circuit.token_cells, circuit._growth_feature_count)
            else:
                expected = existing.shape
            value = np.array(state[key], dtype=dtype, copy=True)
            if value.shape != expected:
                raise ValueError(
                    f"checkpoint array {key} has shape {value.shape}, expected {expected}"
                )
            setattr(circuit, attribute, value)
        if bool(metadata["developmental_state"]):
            developmental_shapes = {
                "_development_cell_order": (circuit.config.reservoir_cells,),
                "_development_active_mask": (circuit.config.reservoir_cells,),
                "_structural_scores": circuit._recurrent_weights_pa.shape,
                "_structural_exposure": circuit._recurrent_weights_pa.shape,
                "_homeostatic_gain": (circuit.config.reservoir_cells,),
            }
            for attribute, (key, dtype) in _DEVELOPMENTAL_ARRAYS.items():
                value = np.array(state[key], dtype=dtype, copy=True)
                expected = developmental_shapes[attribute]
                if value.shape != expected:
                    raise ValueError(
                        f"checkpoint array {key} has shape {value.shape}, expected {expected}"
                    )
                setattr(circuit, attribute, value)
        circuit.active_reservoir_cells = int(metadata["runtime_active_reservoir_cells"])
        if bool(metadata["developmental_state"]):
            if int(np.count_nonzero(circuit._development_active_mask)) != (
                circuit.active_reservoir_cells
            ):
                raise ValueError("checkpoint active mask/count mismatch")
            if not np.array_equal(
                np.sort(circuit._development_cell_order),
                np.arange(circuit.config.reservoir_cells, dtype=np.int32),
            ):
                raise ValueError("checkpoint developmental cell order is not a permutation")
        circuit._bind_native_kernel()
        circuit._growth_examples = int(metadata["growth_examples"])
        circuit._local_updates = int(metadata["local_updates"])
        circuit._learning_frozen = True
    return RecurrentCellLanguageModel(
        tokenizer=tokenizer,
        circuit=circuit,
        training_digest=str(metadata["training_digest"]),
    )
