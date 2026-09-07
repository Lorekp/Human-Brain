"""Rolling128-byte per-call prefix activity with32 exact occupied lag rows.

The first bank evaluates BOS and all31 empty query rows. Each subsequent raw
byte evaluates one new row; the other rows retain their actual fixed activity.
Fresh normalization preserves the frozen full-kernel float32 operation order.
"""

import math

import torch
from torch import Tensor

from humanbrain.neuronal_byte_coincidence import (
    _add,
    encode_symbol_rails,
    validate_symbol_rails,
)
from humanbrain.plastic_memory import _all_finite

WORK_KEYS = (
    "prefix_cache_constructions", "prefix_cache_advances", "prefix_cache_feature_calls",
    "prefix_cache_gate_blocks", "prefix_cache_lag_pairs",
    "prefix_cache_and_cell_evaluations", "prefix_cache_match_cell_evaluations",
    "prefix_cache_threshold_evaluations", "prefix_cache_input_applications",
    "prefix_cache_pair_additions", "prefix_cache_rail_sum_additions",
    "prefix_cache_match_input_cast_values", "prefix_cache_operand_cast_values",
    "prefix_cache_occupancy_rail_reductions", "prefix_cache_occupancy_count_additions",
    "prefix_cache_gate_activity_copy_values", "prefix_cache_reused_activity_values",
    "prefix_cache_shift_copy_values", "prefix_cache_feature_cast_values",
    "prefix_cache_feature_scalings", "prefix_cache_feature_copy_values",
    "prefix_cache_population_products", "prefix_cache_population_square_roots",
    "prefix_cache_population_clamps", "prefix_cache_gain_fills",
    "prefix_cache_gain_divisions", "prefix_cache_held_key_bytes",
    "prefix_cache_held_match_bytes", "prefix_cache_held_count_bytes",
    "prefix_cache_named_allocated_bytes", "prefix_cache_rolling_drops",
)


class RollingPrefixCoincidence:
    """Own one immutable key copy and one evolving prefix's fixed activity.

    Keys are detached CPU bool[N,32,18], N1..512, with raw or empty rows.
    The constructor's numeric key copy prevents caller mutations from changing
    this call's circuit. The caller owns model/tape custody and discards this
    object after its free/teacher call; it is not model or checkpoint state.

    Work counters count actual operations. Reused activity and shift copies
    are separate from evaluated gates. Held-byte counters describe allocations
    made by constructors (summed if work is shared), not a process peak.
    Named allocations exclude validation/encoder internals; the encoder has
    its own existing counters. There is no trainable weight or error input.
    At most128 raw bytes may advance this call. The match bank always describes
    the most recent32 symbols: BOS disappears on advance32; normalization uses
    occupied lags, not the total number of bytes previously emitted.
    """

    @torch.no_grad()
    def __init__(self, keys: Tensor, work: dict | None = None):
        if (
            not isinstance(keys, Tensor) or keys.ndim != 3
            or tuple(keys.shape[1:]) != (32, 18) or not 1 <= keys.shape[0] <= 512
        ):
            raise ValueError("prefix keys require [1..512,32,18]")
        validate_symbol_rails(keys, allow_special=False, work=work)
        self._work = work
        self._used = keys.shape[0]
        self._length = 1  # BOS initially occupies one lag; old lags roll off after32.
        self._raw_count = 0
        self._keys = keys.to(torch.uint8)
        occupied = keys.any(dim=-1)
        self._key_lengths = occupied.sum(dim=-1, dtype=torch.float32)
        del occupied
        self._matches = torch.empty(self._used, 32, 32, dtype=torch.bool)
        token = torch.tensor([257], dtype=torch.int64)
        encoded = encode_symbol_rails(token, work=work)
        query = torch.zeros(32, 18, dtype=torch.uint8)
        bos = encoded.to(torch.uint8)
        query[0].copy_(bos[0])
        del token, encoded, bos
        self._fill(query, self._matches)
        for name, value in {
            "prefix_cache_constructions": 1,
            "prefix_cache_operand_cast_values": self._keys.numel() + 18,
            "prefix_cache_occupancy_rail_reductions": self._used * 32 * 17,
            "prefix_cache_occupancy_count_additions": self._used * 31,
            "prefix_cache_held_key_bytes": self._keys.numel(),
            "prefix_cache_held_match_bytes": self._matches.numel(),
            "prefix_cache_held_count_bytes": self._key_lengths.numel() * 4,
            "prefix_cache_named_allocated_bytes": (
                self._keys.numel() + self._used * 32 + self._used * 4
                + self._matches.numel() + 8 + 18 + query.numel()
            ),
        }.items():
            _add(work, name, value)

    @property
    def raw_count(self) -> int:
        return self._raw_count

    @property
    def occupied_lags(self) -> int:
        return self._length

    def _fill(self, query: Tensor, destination: Tensor) -> None:
        """Evaluate the same two integer threshold layers in128-slot blocks."""
        for first in range(0, self._used, 128):
            last = min(first + 128, self._used)
            currents = query[None, :, None, :] + self._keys[first:last, None, :, :]
            and_activity = currents >= 2
            del currents
            match_inputs = and_activity.to(torch.uint8)
            del and_activity
            match_currents = match_inputs.sum(dim=-1, dtype=torch.uint8)
            del match_inputs
            activity = match_currents >= 9
            del match_currents
            destination[first:last].copy_(activity)
            pairs = activity.numel()
            del activity
            for name, value in {
                "prefix_cache_gate_blocks": 1,
                "prefix_cache_lag_pairs": pairs,
                "prefix_cache_and_cell_evaluations": 18 * pairs,
                "prefix_cache_match_cell_evaluations": pairs,
                "prefix_cache_threshold_evaluations": 19 * pairs,
                "prefix_cache_input_applications": 54 * pairs,
                "prefix_cache_pair_additions": 18 * pairs,
                "prefix_cache_rail_sum_additions": 17 * pairs,
                "prefix_cache_match_input_cast_values": 18 * pairs,
                "prefix_cache_gate_activity_copy_values": pairs,
                "prefix_cache_named_allocated_bytes": 56 * pairs,
            }.items():
                _add(self._work, name, value)

    @torch.no_grad()
    def advance(self, rawbyte: int) -> None:
        """Append one already emitted/offered byte; EOS and BOS are forbidden."""
        if type(rawbyte) is not int or not 0 <= rawbyte <= 255:
            raise ValueError("prefix advancement requires one raw byte")
        if self._raw_count >= 128:
            raise ValueError("prefix cannot contain more than128 raw bytes")
        token = torch.tensor([rawbyte], dtype=torch.int64)
        encoded = encode_symbol_rails(token, work=self._work)
        query = encoded.to(torch.uint8)
        del token, encoded
        # Evaluate before mutating retained activity; no partial shift on a
        # failed gate computation. The full caller's learning is still staged.
        new_row = torch.empty(self._used, 1, 32, dtype=torch.bool)
        self._fill(query, new_row)
        shifted = self._matches[:, :-1].clone()
        self._matches[:, 1:].copy_(shifted)
        del shifted
        self._matches[:, :1].copy_(new_row)
        self._raw_count += 1
        self._length = min(self._raw_count + 1, 32)
        shifted_values = self._used * 31 * 32
        for name, value in {
            "prefix_cache_advances": 1,
            "prefix_cache_rolling_drops": int(self._raw_count >= 32),
            "prefix_cache_operand_cast_values": 18,
            "prefix_cache_reused_activity_values": shifted_values,
            "prefix_cache_shift_copy_values": 2 * shifted_values,
            "prefix_cache_gate_activity_copy_values": new_row.numel(),
            "prefix_cache_named_allocated_bytes": 8 + 18 + new_row.numel() + shifted_values,
        }.items():
            _add(self._work, name, value)

    @torch.no_grad()
    def features(self) -> Tensor:
        """Return a fresh full presynaptic bank, never a recursively rescaled X."""
        features = torch.empty(self._used, 32, 32, dtype=torch.float32)
        length = torch.tensor(self._length, dtype=torch.float32)
        for first in range(0, self._used, 128):
            last = min(first + 128, self._used)
            block = self._matches[first:last].to(torch.float32)
            denominators = length * self._key_lengths[first:last]
            denominators.sqrt_().clamp_min_(1)
            gains = torch.full_like(denominators, math.sqrt(32))
            gains.div_(denominators)
            block.mul_(gains[:, None, None])
            if not _all_finite(block):
                raise ValueError("incremental prefix features must be finite")
            features[first:last].copy_(block)
            del block, denominators, gains
        pairs = features.numel()
        for name, value in {
            "prefix_cache_feature_calls": 1,
            "prefix_cache_feature_cast_values": pairs,
            "prefix_cache_feature_scalings": pairs,
            "prefix_cache_feature_copy_values": pairs,
            "prefix_cache_population_products": self._used,
            "prefix_cache_population_square_roots": self._used,
            "prefix_cache_population_clamps": self._used,
            "prefix_cache_gain_fills": self._used,
            "prefix_cache_gain_divisions": self._used,
            "prefix_cache_named_allocated_bytes": 8 * pairs + 8 * self._used + 4,
        }.items():
            _add(self._work, name, value)
        return features
