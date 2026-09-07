"""Fixed factual comparators and gated copying over every occupied address.

Each relay copies an original float32 potential widened to float64 and its
identity. Contiguous brackets and left ties preserve the lowest original index.
There are no learned parameters, cross-call state or candidate filters.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TournamentLevel:
    """Actual comparator activities and copied outputs, including any odd carry."""

    input_count: int
    pair_currents: Tensor
    left_wins: Tensor
    score_relays: Tensor
    identity_relays: Tensor


@dataclass(frozen=True)
class FactualAddressTournament:
    """Owned original scores and the complete linear-size forward circuit trace."""

    scores: Tensor
    winner_index: int
    winner_score: float
    levels: tuple[TournamentLevel, ...]
    work: dict[str, int]


@torch.no_grad()
def compete_factual(scores: Tensor) -> FactualAddressTournament:
    """Compete detached finite CPU float32 [1..2048] using exact copy relays.

    ``where`` is the comparator-controlled, two-input copy primitive for both
    potential and identity. Logical gated branches are counted separately from
    these fused tensor operations. It never ranks candidates on its own.

    Named bytes count explicit tensor buffers, including the input finite check,
    once per allocation. Peak bytes follow their actual lifetimes and retained
    level receipts. Views, Python objects and framework internals are excluded;
    these counters are not RSS measurements.
    """
    if (
        not isinstance(scores, Tensor)
        or scores.ndim != 1
        or not 1 <= scores.numel() <= 2048
        or scores.dtype != torch.float32
        or scores.device.type != "cpu"
        or scores.requires_grad
        or scores.grad_fn is not None
        or scores.grad is not None
    ):
        raise ValueError("factual scores must be finite detached CPU float32 [1..2048]")

    held_scores = scores.clone()
    n = held_scores.numel()
    finite_activity = torch.isfinite(held_scores)
    all_finite = finite_activity.all()
    if not bool(all_finite.item()):
        raise ValueError("factual scores must be finite detached CPU float32 [1..2048]")
    del finite_activity, all_finite

    current_scores = held_scores.to(torch.float64)
    current_ids = torch.arange(n, dtype=torch.int64, device="cpu")
    work = {
        "factual_tournament_calls": 1,
        "factual_tournament_candidates": n,
        "factual_tournament_input_validation_values": n,
        "factual_tournament_finite_reductions": n - 1,
        "factual_tournament_validation_scalar_reads": 1,
        "factual_tournament_score_clone_values": n,
        "factual_tournament_score_widen_values": n,
        "factual_tournament_identity_initializations": n,
        "factual_tournament_comparator_levels": 0,
        "factual_tournament_relay_levels": 0,
        "factual_tournament_comparators": 0,
        "factual_tournament_signed_comparator_applications": 0,
        "factual_tournament_comparator_subtractions": 0,
        "factual_tournament_thresholds": 0,
        "factual_tournament_logical_control_complements": 0,
        "factual_tournament_logical_gated_payload_branches": 0,
        "factual_tournament_logical_control_fanouts": 0,
        "factual_tournament_potential_relay_outputs": 0,
        "factual_tournament_identity_relay_outputs": 0,
        "factual_tournament_tensor_where_calls": 0,
        "factual_tournament_tensor_where_outputs": 0,
        "factual_tournament_odd_carries": 0,
        "factual_tournament_concatenation_calls": 0,
        "factual_tournament_concatenation_copy_values": 0,
        "factual_tournament_final_scalar_reads": 2,
        "factual_tournament_named_allocated_bytes": 21 * n + 1,
        "factual_tournament_retained_tensor_bytes": 4 * n,
        "factual_tournament_peak_named_live_bytes": max(20 * n, 5 * n + 1),
    }
    levels = []
    width = n
    while width > 1:
        pairs = width // 2
        next_width = pairs + width % 2
        # Views preserve fixed adjacent wiring without copied candidate banks.
        left_scores = current_scores[: 2 * pairs : 2]
        right_scores = current_scores[1 : 2 * pairs : 2]
        left_ids = current_ids[: 2 * pairs : 2]
        right_ids = current_ids[1 : 2 * pairs : 2]
        pair_currents = left_scores - right_scores
        left_wins = pair_currents >= 0
        # Copying preserves the original potential, including signed zero.
        selected_scores = torch.where(left_wins, left_scores, right_scores)
        selected_ids = torch.where(left_wins, left_ids, right_ids)
        extra_bytes = 0
        if width % 2:
            score_relays = torch.cat((selected_scores, current_scores[-1:]))
            identity_relays = torch.cat((selected_ids, current_ids[-1:]))
            extra_bytes = 16 * next_width
            work["factual_tournament_odd_carries"] += 1
            work["factual_tournament_concatenation_calls"] += 2
            work["factual_tournament_concatenation_copy_values"] += 2 * next_width
        else:
            score_relays, identity_relays = selected_scores, selected_ids

        # At this point odd concatenations and their selected inputs coexist.
        initial_lanes = 16 * n if not levels else 0
        live_bytes = (
            work["factual_tournament_retained_tensor_bytes"]
            + initial_lanes
            + 25 * pairs
            + extra_bytes
        )
        work["factual_tournament_peak_named_live_bytes"] = max(
            work["factual_tournament_peak_named_live_bytes"], live_bytes
        )
        work["factual_tournament_named_allocated_bytes"] += 25 * pairs + extra_bytes
        work["factual_tournament_retained_tensor_bytes"] += 9 * pairs + 16 * next_width
        del selected_scores, selected_ids, left_scores, right_scores, left_ids, right_ids
        levels.append(
            TournamentLevel(width, pair_currents, left_wins, score_relays, identity_relays)
        )
        # Previous level outputs remain in the receipt; initial lane banks do not.
        del current_scores, current_ids
        current_scores, current_ids = score_relays, identity_relays
        width = next_width
        work["factual_tournament_comparator_levels"] += 1
        work["factual_tournament_relay_levels"] += 1
        work["factual_tournament_comparators"] += pairs
        work["factual_tournament_signed_comparator_applications"] += 2 * pairs
        work["factual_tournament_comparator_subtractions"] += pairs
        work["factual_tournament_thresholds"] += pairs
        work["factual_tournament_logical_control_complements"] += pairs
        work["factual_tournament_logical_gated_payload_branches"] += 4 * pairs
        work["factual_tournament_logical_control_fanouts"] += 4 * pairs
        work["factual_tournament_potential_relay_outputs"] += next_width
        work["factual_tournament_identity_relay_outputs"] += next_width
        work["factual_tournament_tensor_where_calls"] += 2
        work["factual_tournament_tensor_where_outputs"] += 2 * pairs

    # The final relay already carries the winner; these reads perform no ranking.
    winner_index = int(current_ids[0].item())
    winner_score = float(current_scores[0].item())
    return FactualAddressTournament(held_scores, winner_index, winner_score, tuple(levels), work)
