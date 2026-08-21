"""Frozen artifact format for recurrent human-cell language circuits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from humanbrain.artifacts import _pack_phrases, _unpack_phrases
from humanbrain.cells import ALLEN_GLIF5_CONFIG_SHA256, AllenGlif5Parameters
from humanbrain.readout import SparseCellReadout
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel
from humanbrain.tokenizer import BytePhraseTokenizer

FORMAT = "recurrent-human-cell-checkpoint-v3"
STATE_ENVELOPE_FORMAT = "recurrent-cell-synapse-envelope-v1"

_DEVELOPMENTAL_ARRAYS = {
    "_development_cell_order": ("development_cell_order", np.int32),
    "_development_active_mask": ("development_active_mask", np.bool_),
    "_structural_scores": ("structural_scores", np.float64),
    "_structural_exposure": ("structural_exposure", np.float64),
    "_homeostatic_gain": ("homeostatic_gain", np.float64),
}

_BASE_ARRAYS = {
    "_afferent_targets": ("afferent_targets", np.int32),
    "_recurrent_targets": ("recurrent_targets", np.int32),
    "_recurrent_weights_pa": ("recurrent_weights_pa", np.float64),
    "_output_heterogeneity_pa": ("output_heterogeneity_pa", np.float64),
    "_pacemaker_synapses_pa": ("pacemaker_synapses_pa", np.float64),
}


@dataclass(frozen=True)
class RecurrentCellSynapseState:
    """Readout-agnostic persistent anatomy and learned state for one circuit."""

    metadata: dict[str, object]
    arrays: dict[str, np.ndarray]

    @classmethod
    def capture(cls, circuit: RecurrentCellCircuit) -> RecurrentCellSynapseState:
        if not circuit.learning_frozen:
            raise RuntimeError("freeze recurrent readout learning before checkpointing")
        if circuit.config.readout_learning_rule == "ridge":
            circuit._flush_growth_statistics()
            if circuit._normal_matrix is None or circuit._target_moments is None:
                raise RuntimeError("cannot checkpoint a recurrent circuit before growth")
        elif circuit._normal_matrix is not None or circuit._target_moments is not None:
            raise RuntimeError("local readout must not contain ridge statistics")

        developmental_present = [
            hasattr(circuit, name) for name in _DEVELOPMENTAL_ARRAYS
        ]
        if any(developmental_present) and not all(developmental_present):
            raise RuntimeError("developmental checkpoint state is incomplete")
        developmental_state = all(developmental_present)
        metadata: dict[str, object] = {
            "token_cells": circuit.token_cells,
            "allen_glif5_config_sha256": ALLEN_GLIF5_CONFIG_SHA256,
            "cell_parameters": asdict(circuit.cell_parameters),
            "circuit_config": asdict(circuit.config),
            "growth_examples": circuit._growth_examples,
            "local_updates": circuit.local_update_count,
            "runtime_active_reservoir_cells": circuit.active_reservoir_cells,
            "developmental_state": developmental_state,
        }
        sparse_readout = circuit._sparse_readout
        if circuit.config.readout_learning_rule == "sparse_three_factor":
            if sparse_readout is None:
                raise RuntimeError("sparse readout checkpoint state is missing")
            metadata["sparse_readout"] = sparse_readout.checkpoint_counters()
        elif sparse_readout is not None:
            raise RuntimeError("dense readout must not contain sparse checkpoint state")

        arrays = {
            key: getattr(circuit, attribute)
            for attribute, (key, _) in _BASE_ARRAYS.items()
        }
        if sparse_readout is None:
            if circuit._readout_synapses_pa is None:
                raise RuntimeError("dense readout checkpoint state is missing")
            arrays["readout_synapses_pa"] = circuit._readout_synapses_pa
        else:
            arrays.update(sparse_readout.checkpoint_arrays())
        if circuit.config.readout_learning_rule == "ridge":
            assert circuit._normal_matrix is not None
            assert circuit._target_moments is not None
            arrays["normal_matrix"] = circuit._normal_matrix
            arrays["target_moments"] = circuit._target_moments
        if developmental_state:
            for attribute, (key, _) in _DEVELOPMENTAL_ARRAYS.items():
                arrays[key] = getattr(circuit, attribute)
        return cls(metadata=metadata, arrays=arrays)

    @classmethod
    def from_checkpoint(
        cls,
        metadata: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> RecurrentCellSynapseState:
        if metadata.get("allen_glif5_config_sha256") != ALLEN_GLIF5_CONFIG_SHA256:
            raise ValueError("checkpoint uses a different adaptive Allen human cell")
        config_value = metadata.get("circuit_config")
        if not isinstance(config_value, dict):
            raise ValueError("checkpoint circuit configuration is missing")
        config = RecurrentCircuitConfig(**config_value)
        keys = [key for key, _ in _BASE_ARRAYS.values()]
        if config.readout_learning_rule == "sparse_three_factor":
            keys.extend(
                (
                    "readout_potential_sources",
                    "readout_apposition_eligibility",
                    "readout_contact_sources",
                    "readout_contact_weights_pa",
                    "readout_contact_ages",
                    "readout_output_activity_ema",
                    "readout_target_presentations",
                )
            )
        else:
            keys.append("readout_synapses_pa")
        if config.readout_learning_rule == "ridge":
            keys.extend(("normal_matrix", "target_moments"))
        if bool(metadata.get("developmental_state")):
            keys.extend(key for key, _ in _DEVELOPMENTAL_ARRAYS.values())
        try:
            copied = {key: np.array(arrays[key], copy=True) for key in keys}
        except KeyError as error:
            raise ValueError(f"checkpoint array {error.args[0]} is missing") from error
        state_metadata = {
            key: metadata[key]
            for key in (
                "token_cells",
                "allen_glif5_config_sha256",
                "cell_parameters",
                "circuit_config",
                "growth_examples",
                "local_updates",
                "runtime_active_reservoir_cells",
                "developmental_state",
            )
        }
        if config.readout_learning_rule == "sparse_three_factor":
            state_metadata["sparse_readout"] = metadata.get("sparse_readout")
        return cls(metadata=state_metadata, arrays=copied)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = sha256(encoded)
        for name in sorted(self.arrays):
            value = np.ascontiguousarray(self.arrays[name])
            digest.update(name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()

    def save_enveloped_checkpoint(
        self,
        path: Path,
        envelope: Mapping[str, object],
    ) -> str:
        """Persist this state under caller metadata without exposing array layout."""
        if path.suffix != ".npz":
            raise ValueError("recurrent checkpoint path must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": STATE_ENVELOPE_FORMAT,
            "envelope": dict(envelope),
            "cell_synapse_state": self.metadata,
        }
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                **self.arrays,
            )
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load_enveloped_checkpoint(
        cls,
        path: Path,
    ) -> tuple[dict[str, object], RecurrentCellSynapseState]:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            if metadata.get("format") != STATE_ENVELOPE_FORMAT:
                raise ValueError("unsupported recurrent state envelope")
            envelope = metadata.get("envelope")
            state_metadata = metadata.get("cell_synapse_state")
            if not isinstance(envelope, dict) or not isinstance(state_metadata, dict):
                raise ValueError("recurrent state envelope metadata is malformed")
            state = cls.from_checkpoint(state_metadata, payload)
        return envelope, state

    def restore(self) -> RecurrentCellCircuit:
        metadata = self.metadata
        token_cells = int(metadata["token_cells"])
        circuit = RecurrentCellCircuit(
            token_cells=token_cells,
            config=RecurrentCircuitConfig(**metadata["circuit_config"]),
            cell_parameters=AllenGlif5Parameters(**metadata["cell_parameters"]),
        )
        bindings = dict(_BASE_ARRAYS)
        if circuit.config.readout_learning_rule != "sparse_three_factor":
            bindings["_readout_synapses_pa"] = ("readout_synapses_pa", np.float64)
        if circuit.config.readout_learning_rule == "ridge":
            bindings["_normal_matrix"] = ("normal_matrix", np.float64)
            bindings["_target_moments"] = ("target_moments", np.float64)
        for attribute, (key, dtype) in bindings.items():
            existing = getattr(circuit, attribute)
            if existing is None and attribute == "_normal_matrix":
                expected = (circuit._growth_feature_count, circuit._growth_feature_count)
            elif existing is None and attribute == "_target_moments":
                expected = (circuit.token_cells, circuit._growth_feature_count)
            else:
                expected = existing.shape
            value = np.array(self.arrays[key], dtype=dtype, copy=True, order="C")
            if value.shape != expected:
                raise ValueError(
                    f"checkpoint array {key} has shape {value.shape}, expected {expected}"
                )
            setattr(circuit, attribute, value)

        if circuit.config.readout_learning_rule == "sparse_three_factor":
            sparse_counters = metadata.get("sparse_readout")
            if not isinstance(sparse_counters, dict):
                raise ValueError("sparse checkpoint counters are missing")
            circuit._sparse_readout = SparseCellReadout.from_checkpoint(
                output_cells=circuit.token_cells,
                reservoir_cells=circuit.config.reservoir_cells,
                fan_in=circuit.config.readout_fan_in,
                reservoir_eligibility=circuit._eligibility,
                output_heterogeneity_pa=circuit._output_heterogeneity_pa,
                pacemaker_synapses_pa=circuit._pacemaker_synapses_pa,
                arrays=self.arrays,
                counters=sparse_counters,
            )
            if circuit._sparse_readout.applied_updates != int(metadata["local_updates"]):
                raise ValueError("sparse checkpoint update counters disagree")

        if bool(metadata["developmental_state"]):
            developmental_shapes = {
                "_development_cell_order": (circuit.config.reservoir_cells,),
                "_development_active_mask": (circuit.config.reservoir_cells,),
                "_structural_scores": circuit._recurrent_weights_pa.shape,
                "_structural_exposure": circuit._recurrent_weights_pa.shape,
                "_homeostatic_gain": (circuit.config.reservoir_cells,),
            }
            for attribute, (key, dtype) in _DEVELOPMENTAL_ARRAYS.items():
                value = np.array(self.arrays[key], dtype=dtype, copy=True, order="C")
                expected = developmental_shapes[attribute]
                if value.shape != expected:
                    raise ValueError(
                        f"checkpoint array {key} has shape {value.shape}, expected {expected}"
                    )
                setattr(circuit, attribute, value)
        circuit.active_reservoir_cells = int(metadata["runtime_active_reservoir_cells"])
        if bool(metadata["developmental_state"]):
            if (
                int(np.count_nonzero(circuit._development_active_mask))
                != circuit.active_reservoir_cells
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
        return circuit


def save_recurrent_checkpoint(model: RecurrentCellLanguageModel, path: Path) -> str:
    if path.suffix != ".npz":
        raise ValueError("recurrent checkpoint path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit = model._circuit
    cell_synapse_state = RecurrentCellSynapseState.capture(circuit)
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
        circuit_state = RecurrentCellSynapseState.from_checkpoint(metadata, state)
        circuit = circuit_state.restore()
    return RecurrentCellLanguageModel(
        tokenizer=tokenizer,
        circuit=circuit,
        training_digest=str(metadata["training_digest"]),
    )
