"""Stateless symbol rails and two fixed coincidence threshold layers.

All lag pairs of every occupied address are evaluated before learned address
synapses receive the resulting activity. Binary storage and uint8 currents are
exact representations of these fixed cells, not a host symbol-equality lookup.
"""

import math

import torch
from torch import Tensor

from humanbrain.plastic_memory import _all_finite

WORK_KEYS = (
    "rail_encode_symbols", "rail_encode_bit_positions", "rail_encode_bit_shifts",
    "rail_encode_bit_masks", "rail_encode_boolean_cast_values", "rail_encode_complements",
    "rail_encode_output_values", "rail_encode_named_allocated_bytes",
    "rail_validation_rows", "rail_validation_complementary_pair_checks",
    "rail_validation_special_policy_rows",
    "coincidence_feature_calls", "coincidence_slots", "coincidence_lag_pairs",
    "coincidence_and_cell_evaluations", "coincidence_match_cell_evaluations",
    "coincidence_threshold_evaluations", "coincidence_and_input_applications",
    "coincidence_match_input_applications", "coincidence_input_applications",
    "coincidence_pair_additions", "coincidence_rail_sum_additions",
    "coincidence_operand_cast_values", "coincidence_match_input_cast_values",
    "coincidence_occupancy_rail_reductions", "coincidence_occupancy_count_additions",
    "coincidence_population_products", "coincidence_population_square_roots",
    "coincidence_population_clamps", "coincidence_gain_fills", "coincidence_gain_divisions",
    "coincidence_feature_cast_values", "coincidence_feature_scalings",
    "coincidence_named_operand_cast_bytes", "coincidence_named_rail_current_bytes",
    "coincidence_named_and_activity_bytes", "coincidence_named_match_input_bytes",
    "coincidence_named_match_current_bytes", "coincidence_named_match_activity_bytes",
    "coincidence_named_feature_bytes", "coincidence_named_occupancy_bytes",
    "coincidence_named_occupied_count_bytes", "coincidence_named_denominator_bytes",
    "coincidence_named_gain_bytes", "coincidence_named_allocated_bytes",
)


def _add(work: dict | None, name: str, value: int) -> None:
    if work is not None:
        work[name] = work.get(name, 0) + value


def _detached_cpu(value: Tensor, dtype: torch.dtype, name: str) -> None:
    if (
        not isinstance(value, Tensor) or value.dtype != dtype
        or value.device.type != "cpu" or value.requires_grad
        or value.grad_fn is not None or value.grad is not None
    ):
        raise ValueError(f"{name} must be detached CPU {dtype}")


@torch.no_grad()
def encode_symbol_rails(tokens: Tensor, work: dict | None = None) -> Tensor:
    """Fan out IDs 0..257 into nine low-bit-first (zero, one) rail pairs.

    Bytes are 0..255, EOS is 256 and BOS is 257. No token denotes the empty
    population: empty lag rows are separately initialized to eighteen zeros.
    There is no persistent code table or trainable encoding parameter.
    """
    _detached_cpu(tokens, torch.int64, "symbol IDs")
    if bool(((tokens < 0) | (tokens > 257)).any()):
        raise ValueError("symbol IDs must be raw bytes, EOS or BOS")
    positions = torch.arange(9, dtype=torch.int64, device="cpu")
    bits = torch.bitwise_right_shift(tokens[..., None], positions)
    bits.bitwise_and_(1)
    ones = bits.to(torch.bool)
    del bits
    zeros = ~ones
    rails = torch.stack((zeros, ones), dim=-1).reshape(*tokens.shape, 18)
    count = tokens.numel()
    for name, value in {
        "rail_encode_symbols": count,
        "rail_encode_bit_positions": 9,
        "rail_encode_bit_shifts": 9 * count,
        "rail_encode_bit_masks": 9 * count,
        "rail_encode_boolean_cast_values": 9 * count,
        "rail_encode_complements": 9 * count,
        "rail_encode_output_values": 18 * count,
        "rail_encode_named_allocated_bytes": 108 * count + 72,
    }.items():
        _add(work, name, value)
    return rails


@torch.no_grad()
def validate_symbol_rails(
    rails: Tensor, *, allow_special: bool = True, work: dict | None = None,
) -> None:
    """Validate empty or canonical symbol rows, without pairwise retrieval.

    Every occupied row has exactly one active rail per bit. High bit one allows
    only lower bits zero (EOS) or bit zero one (BOS). Raw-key callers pass
    allow_special=False; the generic gate may use both specials in truth QA.
    """
    _detached_cpu(rails, torch.bool, "symbol rails")
    if rails.ndim < 1 or rails.shape[-1] != 18 or type(allow_special) is not bool:
        raise ValueError("symbol rails require [...,18] and an explicit boolean policy")
    zero_rails, one_rails = rails[..., 0::2], rails[..., 1::2]
    occupied = rails.any(dim=-1)
    complete = (zero_rails ^ one_rails).all(dim=-1)
    if bool((occupied & ~complete).any()):
        raise ValueError("occupied symbols require one active rail per bit")
    high = one_rails[..., 8]
    if allow_special:
        if bool((high & one_rails[..., 1:8].any(dim=-1)).any()):
            raise ValueError("symbol rails encode an undeclared symbol above BOS")
    elif bool(high.any()):
        raise ValueError("stored raw prefix rails cannot contain EOS or BOS")
    count = rails.numel() // 18
    _add(work, "rail_validation_rows", count)
    _add(work, "rail_validation_complementary_pair_checks", 9 * count)
    _add(work, "rail_validation_special_policy_rows", count)


@torch.no_grad()
def coincidence_features(
    query_rails: Tensor, key_rails: Tensor, work: dict | None = None,
) -> Tensor:
    """Evaluate all two-layer coincidences, then scale their actual activity.

    Query has [32,18] rails; keys have [N,32,18], N=1..128. Specials are valid
    here; the memory's separate raw-prefix contract excludes them from keys.
    A nonempty query is required. Float32 X equals binary M times a float32
    gain: float32(sqrt(32)) divided by sqrt(float32(Lq*Lk)). The denominator is
    clamped to one for empty keys, whose actual match bank is already zero.

    Named allocation counters sum explicitly created tensors once, not their
    simultaneous RSS or validation/allocator temporaries. No activity survives
    this call except the returned normalized presynaptic bank. The caller may
    hold that bank through its current local teacher; no cross-call cache exists.
    """
    if (
        not isinstance(query_rails, Tensor) or tuple(query_rails.shape) != (32, 18)
        or not isinstance(key_rails, Tensor) or key_rails.ndim != 3
        or tuple(key_rails.shape[1:]) != (32, 18) or not 1 <= key_rails.shape[0] <= 128
    ):
        raise ValueError("coincidence requires query[32,18] and keys[1..128,32,18]")
    validate_symbol_rails(query_rails, work=work)
    validate_symbol_rails(key_rails, work=work)
    n = key_rails.shape[0]
    pairs = n * 32 * 32
    query_occupied = query_rails.any(dim=-1)
    key_occupied = key_rails.any(dim=-1)
    query_length = query_occupied.sum(dtype=torch.float32)
    key_lengths = key_occupied.sum(dim=-1, dtype=torch.float32)
    del query_occupied, key_occupied
    if int(query_length.item()) == 0:
        raise ValueError("coincidence query must contain at least one actual symbol")

    # Both operands must be numeric: bool addition would lose the current two.
    query8 = query_rails.to(torch.uint8)
    keys8 = key_rails.to(torch.uint8)
    currents = query8[None, :, None, :] + keys8[:, None, :, :]
    and_activity = currents >= 2  # H(current - 1.5), exact for current 0,1,2.
    del currents, query8, keys8
    match_inputs = and_activity.to(torch.uint8)
    del and_activity
    match_currents = match_inputs.sum(dim=-1, dtype=torch.uint8)
    del match_inputs
    match_activity = match_currents >= 9  # H(current - 8.5).
    del match_currents
    features = match_activity.to(torch.float32)
    del match_activity

    denominators = query_length * key_lengths
    denominators.sqrt_().clamp_min_(1)
    gains = torch.full_like(denominators, math.sqrt(32))
    gains.div_(denominators)
    features.mul_(gains[:, None, None])
    if not _all_finite(features):
        raise ValueError("normalized coincidence activity must be finite")

    for name, value in {
        "coincidence_feature_calls": 1,
        "coincidence_slots": n,
        "coincidence_lag_pairs": pairs,
        "coincidence_and_cell_evaluations": 18 * pairs,
        "coincidence_match_cell_evaluations": pairs,
        "coincidence_threshold_evaluations": 19 * pairs,
        "coincidence_and_input_applications": 36 * pairs,
        "coincidence_match_input_applications": 18 * pairs,
        "coincidence_input_applications": 54 * pairs,
        "coincidence_pair_additions": 18 * pairs,
        "coincidence_rail_sum_additions": 17 * pairs,
        "coincidence_operand_cast_values": 576 * (n + 1),
        "coincidence_match_input_cast_values": 18 * pairs,
        "coincidence_occupancy_rail_reductions": 544 * (n + 1),
        "coincidence_occupancy_count_additions": 31 * (n + 1),
        "coincidence_population_products": n,
        "coincidence_population_square_roots": n,
        "coincidence_population_clamps": n,
        "coincidence_gain_fills": n,
        "coincidence_gain_divisions": n,
        "coincidence_feature_cast_values": pairs,
        "coincidence_feature_scalings": pairs,
        "coincidence_named_operand_cast_bytes": 576 * (n + 1),
        "coincidence_named_rail_current_bytes": 18 * pairs,
        "coincidence_named_and_activity_bytes": 18 * pairs,
        "coincidence_named_match_input_bytes": 18 * pairs,
        "coincidence_named_match_current_bytes": pairs,
        "coincidence_named_match_activity_bytes": pairs,
        "coincidence_named_feature_bytes": 4 * pairs,
        "coincidence_named_occupancy_bytes": 32 * (n + 1),
        "coincidence_named_occupied_count_bytes": 4 * (n + 1),
        "coincidence_named_denominator_bytes": 4 * n,
        "coincidence_named_gain_bytes": 4 * n,
        "coincidence_named_allocated_bytes": 60 * pairs + 612 * (n + 1) + 8 * n,
    }.items():
        _add(work, name, value)
    return features
