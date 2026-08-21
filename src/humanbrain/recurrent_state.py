"""Neutral serialized-state container for recurrent cell circuits."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

STATE_ENVELOPE_FORMAT = "recurrent-cell-synapse-envelope-v1"


@dataclass(frozen=True)
class RecurrentCellSynapseState:
    """Opaque persistent metadata and arrays owned by a recurrent circuit."""

    metadata: dict[str, object]
    arrays: dict[str, np.ndarray]

    @classmethod
    def copy_payload(
        cls,
        metadata: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
        array_names: Iterable[str],
    ) -> RecurrentCellSynapseState:
        """Copy a circuit-selected payload out of an artifact container."""
        try:
            copied = {name: np.array(arrays[name], copy=True, order="C") for name in array_names}
        except KeyError as error:
            raise ValueError(f"checkpoint array {error.args[0]} is missing") from error
        metadata_copy = json.loads(json.dumps(dict(metadata), allow_nan=False))
        return cls(metadata=metadata_copy, arrays=copied)

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
        """Persist this opaque state under caller-owned metadata."""
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
            state = cls.copy_payload(
                state_metadata,
                payload,
                (name for name in payload.files if name != "metadata"),
            )
        return envelope, state


__all__ = ["RecurrentCellSynapseState"]
