"""Two fixed threshold layers and a sparse identity readout over all addresses.

Pair order is the row-major upper triangle i<j. Exact score ties favor the
lower address index. There are no learned weights or retained cross-call state.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from humanbrain.plastic_memory import _all_finite


@dataclass(frozen=True)
class AddressCompetition:
    """Actual forward activities; scores own their storage independently of input."""

    scores: Tensor
    pair_currents: Tensor
    comparator_activity: Tensor
    winner_currents: Tensor
    winner_activity: Tensor
    winner_index: int
    work: dict[str, int]


@torch.no_grad()
def compete(scores: Tensor) -> AddressCompetition:
    """Compete finite CPU float32 scores in exactly two threshold stages.

    Fixed +/-1 wiring uses one subtraction per pair and two signed indexed
    accumulations per comparator. Work records both these actual operations and
    the logical signed synaptic applications. Named allocation bytes include
    each explicitly created tensor once, not allocator/validation temporaries
    or an assertion that all arrays are simultaneously live.
    """
    if (
        not isinstance(scores, Tensor) or scores.ndim != 1 or not 1 <= scores.numel() <= 128
        or scores.dtype != torch.float32 or scores.device.type != "cpu"
        or scores.requires_grad or scores.grad_fn is not None or scores.grad is not None
        or not _all_finite(scores)
    ):
        raise ValueError("address scores must be finite detached CPU float32 [1..128]")

    held_scores = scores.clone()
    n = held_scores.numel()
    pairs = n * (n - 1) // 2
    scores64 = held_scores.to(torch.float64)
    indices = torch.triu_indices(n, n, offset=1, dtype=torch.int64, device="cpu")
    left = scores64.index_select(0, indices[0])
    right = scores64.index_select(0, indices[1])
    pair_currents = left - right
    if not _all_finite(pair_currents):
        raise ValueError("address comparator currents must be finite")
    comparator_activity = pair_currents >= 0
    del left, right, scores64

    positions = torch.arange(n, dtype=torch.int64, device="cpu")
    winner_currents = positions.to(torch.float64)
    comparator64 = comparator_activity.to(torch.float64)
    winner_currents.index_add_(0, indices[0], comparator64)
    winner_currents.index_add_(0, indices[1], comparator64, alpha=-1)
    winner_currents.sub_(n - 1)
    if not _all_finite(winner_currents):
        raise ValueError("address winner currents must be finite")
    winner_activity = winner_currents >= 0
    winner_count = winner_activity.sum()
    if int(winner_count.item()) != 1:
        raise ValueError("address competition must produce exactly one active winner")

    # This fixed weighted readout represents the already active winner neuron;
    # it does not perform an additional ranking/selection of input scores.
    identity_activity = winner_activity.to(torch.int64)
    identity_products = positions * identity_activity
    cursor = identity_products.sum()
    winner_index = int(cursor.item())
    if not 0 <= winner_index < n:
        raise ValueError("address identity readout exceeds the occupied population")

    work = {
        "competition_candidates": n,
        "competition_pairs": pairs,
        "competition_threshold_layers": 2,
        "competition_comparator_neurons": pairs,
        "competition_winner_neurons": n,
        "competition_comparator_synapse_applications": 2 * pairs,
        "competition_winner_synapse_applications": 2 * pairs,
        "competition_comparator_subtractions": pairs,
        "competition_winner_accumulations": 2 * pairs,
        "competition_winner_bias_values": n,
        "competition_winner_threshold_subtractions": n,
        "competition_threshold_evaluations": pairs + n,
        "competition_identity_synapse_applications": n,
        "competition_identity_reductions": n - 1,
        "competition_winner_count_reductions": n - 1,
        "competition_index_values_constructed": 2 * pairs + n,
        "competition_receipt_tensor_bytes": 13 * n + 9 * pairs,
        "competition_named_allocated_tensor_bytes": 45 * n + 49 * pairs + 16,
        "competition_pair_index_bytes": 16 * pairs,
        "competition_score_gather_bytes": 16 * pairs,
    }
    return AddressCompetition(
        held_scores, pair_currents, comparator_activity, winner_currents,
        winner_activity, winner_index, work,
    )
