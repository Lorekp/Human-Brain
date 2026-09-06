"""Fixed mixed context cells preceding local byte/EOS output synapses.

No teacher, address choice or learned weight enters this module. Every supplied
candidate takes the same batched fixed wiring. The caller owns raw query/prefix
construction and the actual memory context; this module checks their geometry
and causal symbol layout before evaluating threshold currents.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from humanbrain.neuronal_byte_coincidence import validate_symbol_rails

INPUT_WIDTH = 2051
OUTPUT_WIDTH = 292
CELL_COUNT = 256
FAN_IN = 32
WIRING_BYTES = CELL_COUNT * FAN_IN * (8 + 1)
BLOCKS = ((0, 576, 8), (576, 306, 8), (882, 1152, 12), (2034, 17, 4))
WORK_KEYS = (
    "output_context_calls", "output_context_candidates",
    "output_context_fixed_wiring_bytes_hashed",
    "output_context_validation_rail_rows",
    "output_context_input_values", "output_context_input_copy_values",
    "output_context_position_zero_fills", "output_context_position_fills",
    "output_context_current_consistency_values", "output_context_afferent_gathers",
    "output_context_signed_input_applications", "output_context_signed_cast_values",
    "output_context_current_reductions", "output_context_threshold_evaluations",
    "output_context_feature_cast_values", "output_context_feature_copy_values",
    "output_context_constant_fills", "output_context_norm_squares",
    "output_context_norm_reductions", "output_context_norm_square_roots",
    "output_context_feature_divisions", "output_context_feature_scalings",
    "output_context_named_input_bytes", "output_context_named_position_bytes",
    "output_context_named_gather_bytes", "output_context_named_signed_cast_bytes",
    "output_context_named_signed_product_bytes", "output_context_named_current_bytes",
    "output_context_named_threshold_bytes", "output_context_named_feature_bytes",
    "output_context_named_feature_square_bytes", "output_context_named_norm_bytes",
    "output_context_named_normalized_bytes", "output_context_named_allocated_bytes",
)


def _add(work: dict | None, name: str, count: int) -> None:
    if work is not None:
        work[name] = work.get(name, 0) + count


def _tensor(value: Tensor, shape: tuple, dtype: torch.dtype, name: str) -> None:
    if (not isinstance(value, Tensor) or tuple(value.shape) != shape
            or value.dtype != dtype or value.device.type != "cpu"
            or value.requires_grad or value.grad_fn is not None or value.grad is not None):
        raise ValueError(f"{name} must retain detached CPU {dtype} {shape}")


def _fingerprint(indices: Tensor, signs: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(indices.contiguous().numpy().tobytes())
    digest.update(signs.contiguous().numpy().tobytes())
    return digest.hexdigest()


class RequestConditionedOutputCells(nn.Module):
    """256 fixed threshold cells and 36 direct features, with energy32.

    Fixed wiring samples each input group without replacement per cell. There
    are no trainable parameters and no state changes during forward evaluation.
    Only indices/signs are checkpoint buffers; their constructor fingerprint
    rejects a different wiring even when shapes and value ranges match.
    """

    def __init__(self, *, seed: int = 733) -> None:
        super().__init__()
        if type(seed) is not int:
            raise ValueError("context wiring seed must be an integer")
        self.seed = seed
        generator = torch.Generator(device="cpu").manual_seed(seed + 104729)
        indices = torch.empty(CELL_COUNT, FAN_IN, dtype=torch.int64)
        signs = torch.empty(CELL_COUNT, FAN_IN, dtype=torch.int8)
        for row in range(CELL_COUNT):
            column = 0
            for offset, width, count in BLOCKS:
                indices[row, column:column + count].copy_(
                    torch.randperm(width, generator=generator)[:count] + offset)
                column += count
            signs[row].copy_(
                (torch.randint(2, (FAN_IN,), generator=generator) * 2 - 1).to(torch.int8))
        self.register_buffer("indices", indices)
        self.register_buffer("signs", signs)
        self._fixed_sha256 = _fingerprint(indices, signs)

    def get_extra_state(self) -> dict:
        return {
            "schema": "request-conditioned-output-fixed-cells-v1",
            "seed": self.seed, "seed_offset": 104729,
            "input_width": INPUT_WIDTH, "cells": CELL_COUNT, "fan_in": FAN_IN,
            "groups": [list(group) for group in BLOCKS],
            "wiring": "percell four randperm subsets then32 iid signs; private CPU generator",
            "wiring_sha256": self._fixed_sha256,
            "current": "signed int8 contributions; int32 sum; threshold current>=0",
            "features": "threshold256,currentrails18,position17,constant1; float64 energy32",
            "query": "beginning-aligned raw bytes then EOS; zero padding32positions",
            "prefix": "BOS then exactly position earlier raw outputs; zero padding17positions",
            "context": "32prior raw/empty then32forward raw/EOS/empty; current is row32",
            "publication": "no learned parameters or forward mutation",
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("fixed output context metadata mismatch")

    def validate_fixed_state(self, state: Mapping, prefix: str = "") -> None:
        """Prevalidate incoming buffers before an owning learner publishes any state."""
        if not isinstance(state, Mapping) or not isinstance(prefix, str):
            raise ValueError("fixed context state must be a mapping with a string prefix")
        indices, signs = state.get(prefix + "indices"), state.get(prefix + "signs")
        _tensor(indices, (CELL_COUNT, FAN_IN), torch.int64, "fixed indices")
        _tensor(signs, (CELL_COUNT, FAN_IN), torch.int8, "fixed signs")
        if (_fingerprint(indices, signs) != self._fixed_sha256
                or state.get(prefix + "_extra_state") != self.get_extra_state()):
            raise ValueError("fixed output context wiring identity mismatch")

    def check_parameters(self) -> None:
        self.validate_fixed_state({
            "indices": self.indices, "signs": self.signs,
            "_extra_state": self.get_extra_state(),
        })

    def load_state_dict(self, state_dict: Mapping, strict: bool = True, assign: bool = False):
        if (not strict or assign or not isinstance(state_dict, Mapping)
                or set(state_dict) != {"indices", "signs", "_extra_state"}):
            raise ValueError("fixed context loading requires exact strict in-place state")
        self.validate_fixed_state(state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=False)

    @torch.no_grad()
    def forward(
        self, query_rails: Tensor, prefix_rails: Tensor, context_rails: Tensor,
        current_rails: Tensor, position: int, work: dict | None = None,
    ) -> Tensor:
        """Evaluate every supplied candidate; return detached float64 [N,292]."""
        self.check_parameters()
        if type(position) is not int or not 0 <= position <= 16:
            raise ValueError("output position must be0..16")
        if (not isinstance(context_rails, Tensor) or context_rails.ndim != 3
                or not 1 <= context_rails.shape[0] <= 512):
            raise ValueError("context must contain1..512 candidates")
        n = context_rails.shape[0]
        _tensor(query_rails, (32, 18), torch.bool, "query rails")
        _tensor(prefix_rails, (17, 18), torch.bool, "prefix rails")
        _tensor(context_rails, (n, 64, 18), torch.bool, "context rails")
        _tensor(current_rails, (n, 18), torch.bool, "current rails")
        for rails in (query_rails, prefix_rails, context_rails, current_rails):
            validate_symbol_rails(rails)
        query_occupied = query_rails.any(dim=1)
        length = int(query_occupied.sum())
        if (length < 1 or not bool(query_occupied[:length].all())
                or bool(query_occupied[length:].any())
                or bool(query_rails[:length - 1, 17].any())
                or not bool(query_rails[length - 1, 17])
                or bool(query_rails[length - 1, 1])):
            raise ValueError("query must contain contiguous raw bytes then exactly one EOS")
        prefix_occupied = prefix_rails.any(dim=1)
        if (not bool(prefix_occupied[:position + 1].all())
                or bool(prefix_occupied[position + 1:].any())
                or not bool(prefix_rails[0, 17] & prefix_rails[0, 1])
                or bool(prefix_rails[1:position + 1, 17].any())):
            raise ValueError("prefix must contain BOS and exactly the earlier raw outputs")
        if (bool((context_rails[..., 17] & context_rails[..., 1]).any())
                or bool((current_rails[..., 17] & current_rails[..., 1]).any())):
            raise ValueError("memory context/current cannot contain BOS")
        if bool(context_rails[:, :32, 17].any()):
            raise ValueError("prior context must contain raw bytes or empty rows only")
        if not torch.equal(current_rails, context_rails[:, 32]):
            raise ValueError("current rails must equal the first forward context row")
        del query_occupied, prefix_occupied

        pos = torch.zeros(17, dtype=torch.bool)
        pos[position] = True
        inputs = torch.empty(n, INPUT_WIDTH, dtype=torch.bool)
        inputs[:, :576].copy_(query_rails.reshape(1, 576))
        inputs[:, 576:882].copy_(prefix_rails.reshape(1, 306))
        inputs[:, 882:2034].copy_(context_rails.reshape(n, 1152))
        inputs[:, 2034:].copy_(pos)
        gathered = inputs.index_select(1, self.indices.reshape(-1)).reshape(n, CELL_COUNT, FAN_IN)
        del inputs
        signed = gathered.to(torch.int8)
        del gathered
        products = signed * self.signs
        del signed
        currents = products.sum(dim=2, dtype=torch.int32)
        del products
        activity = currents >= 0
        del currents
        features = torch.empty(n, OUTPUT_WIDTH, dtype=torch.float64)
        features[:, :256].copy_(activity)
        del activity
        features[:, 256:274].copy_(current_rails)
        features[:, 274:291].copy_(pos)
        features[:, 291] = 1
        del pos
        squares = features.square()
        norm_squared = squares.sum(dim=1, keepdim=True)
        del squares
        norm = norm_squared.sqrt()
        del norm_squared
        normalized = features / norm
        del features, norm
        normalized.mul_(math.sqrt(32))
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("normalized output context must be finite")

        counts = {
            "calls": 1, "candidates": n, "fixed_wiring_bytes_hashed": WIRING_BYTES,
            "validation_rail_rows": 49 + 65 * n,
            "input_values": n * INPUT_WIDTH, "input_copy_values": n * INPUT_WIDTH,
            "position_zero_fills": 17, "position_fills": 1,
            "current_consistency_values": n * 18,
            "afferent_gathers": n * CELL_COUNT * FAN_IN,
            "signed_input_applications": n * CELL_COUNT * FAN_IN,
            "signed_cast_values": n * CELL_COUNT * FAN_IN,
            "current_reductions": n * CELL_COUNT * (FAN_IN - 1),
            "threshold_evaluations": n * CELL_COUNT,
            "feature_cast_values": n * 291, "feature_copy_values": n * 291,
            "constant_fills": n, "norm_squares": n * OUTPUT_WIDTH,
            "norm_reductions": n * (OUTPUT_WIDTH - 1), "norm_square_roots": n,
            "feature_divisions": n * OUTPUT_WIDTH, "feature_scalings": n * OUTPUT_WIDTH,
            "named_input_bytes": n * INPUT_WIDTH, "named_position_bytes": 17,
            "named_gather_bytes": n * CELL_COUNT * FAN_IN,
            "named_signed_cast_bytes": n * CELL_COUNT * FAN_IN,
            "named_signed_product_bytes": n * CELL_COUNT * FAN_IN,
            "named_current_bytes": n * CELL_COUNT * 4,
            "named_threshold_bytes": n * CELL_COUNT,
            "named_feature_bytes": n * OUTPUT_WIDTH * 8,
            "named_feature_square_bytes": n * OUTPUT_WIDTH * 8,
            "named_norm_bytes": n * 8 * 2,
            "named_normalized_bytes": n * OUTPUT_WIDTH * 8,
        }
        counts["named_allocated_bytes"] = sum(
            value for key, value in counts.items() if key.startswith("named_"))
        for name, count in counts.items():
            _add(work, "output_context_" + name, count)
        return normalized

    def accounting(self) -> dict:
        return {
            "fixed_threshold_cells": CELL_COUNT, "fixed_afferents": CELL_COUNT * FAN_IN,
            "fixed_wiring_tensor_bytes": WIRING_BYTES, "learned_parameters": 0,
            "constructor_randperm_calls": CELL_COUNT * len(BLOCKS),
            "constructor_randperm_output_values": CELL_COUNT * INPUT_WIDTH,
            "constructor_randint_calls": CELL_COUNT,
            "constructor_sign_draws": CELL_COUNT * FAN_IN,
            "output_features": OUTPUT_WIDTH, "maximum_candidates": 512,
            "maximum_gather_or_signed_tensor_bytes": 512 * CELL_COUNT * FAN_IN,
            "maximum_current_tensor_bytes": 512 * CELL_COUNT * 4,
            "maximum_returned_feature_bytes": 512 * OUTPUT_WIDTH * 8,
            "named_allocation_scope": (
                "cumulative forward tensors only; excludes caller inputs, validation scratch, "
                "wiring hash byte copies and constructor RNG scratch; not peak RSS"),
            "history": "no call cache or mutable history; caller supplies causal prefix",
        }
