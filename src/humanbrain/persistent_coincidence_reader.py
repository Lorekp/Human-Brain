"""512-slot persistent recall; all addresses compete and learn locally.

Query history remains32 raw/EOS lag rows. This expands literal storage, not
instruction length, action geometry, semantic compression or learned eviction.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_EOS, ByteGeneration
from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner
from humanbrain.competitive_local_memory_learner import CompetitiveSequencePrediction
from humanbrain.local_memory_learner import SequenceQuery, _finite
from humanbrain.neuronal_address_competition import AddressCompetition
from humanbrain.neuronal_byte_coincidence import coincidence_features, encode_symbol_rails
from humanbrain.neuronal_sequence_memory import TapeState
from humanbrain.persistent_coincidence_memory import PersistentCoincidenceMemory512
from humanbrain.plastic_memory import _all_finite


@torch.no_grad()
def compete512(scores: Tensor) -> AddressCompetition:
    """Compete finite CPU float32 scores in exactly two threshold stages.

    Fixed +/-1 wiring uses one subtraction per pair and two signed indexed
    accumulations per comparator. Work records both these actual operations and
    the logical signed synaptic applications. Named allocation bytes include
    each explicitly created tensor once, not allocator/validation temporaries
    or an assertion that all arrays are simultaneously live.
    """
    if (
        not isinstance(scores, Tensor) or scores.ndim != 1 or not 1 <= scores.numel() <= 512
        or scores.dtype != torch.float32 or scores.device.type != "cpu"
        or scores.requires_grad or scores.grad_fn is not None or scores.grad is not None
        or not _all_finite(scores)
    ):
        raise ValueError("address scores must be finite detached CPU float32 [1..512]")

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


class PersistentCoincidenceReader512(CoincidenceLocalMemoryLearner):
    """Same32x32 synapses/codec/teacher with one caller-owned512-slot tape."""

    WORK_KEYS = (*CoincidenceLocalMemoryLearner.WORK_KEYS,
        "persistent_feature_block_count", "persistent_feature_assembly_bytes",
        "persistent_feature_copy_values",
    )

    def __init__(self, *, mode: str = "learned", seed: int = 0) -> None:
        super().__init__(mode=mode, seed=seed)
        self.memory = PersistentCoincidenceMemory512()

    def get_extra_state(self) -> dict:
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "persistent-coincidence-reader-512-v1", "capacity": 512,
            "memory": (
                "append-only bool[512,32,18] preceding raw-byte rails; raw sensory/EOS "
                "and forward absorbing adjacency; no reset/eviction/query-based routing"
            ),
            "address_activity": (
                "all occupied slots in contiguous blocks of at most128 through frozen "
                "coincidence gates; assemble full float32[N,32,32] before score; "
                "retain allX through unchanged direct local teacher"
            ),
            "competition_capacity": 512,
            "query_limit_scope": "last31 raw bytes plus EOS; no longer-instruction claim",
            "feature_block_size": 128,
        })
        return metadata

    @torch.no_grad()
    def _query(self, query: bytes, state: TapeState) -> SequenceQuery:
        self._check_parameters()
        self._check_state(state)
        if not isinstance(query, bytes):
            raise ValueError("query must be raw bytes")
        work = dict.fromkeys(self.WORK_KEYS, 0)
        tokens = torch.tensor([*query, BYTE_EOS], dtype=torch.int64, device="cpu")
        rails = encode_symbol_rails(tokens, work=work)
        lags = torch.zeros(32, 18, dtype=torch.bool, device="cpu")
        for inputs in rails.unbind(0):
            # Every raw byte and EOS use the same forward lag wiring. No task
            # field, extracted key or teacher affects encoding or candidate scope.
            lags[1:].copy_(lags[:-1].clone())
            lags[0].copy_(inputs)
        del rails, tokens, inputs
        activity = lags.reshape(1, 576)
        if state.used <= 128:
            features = coincidence_features(lags, state.keys[:state.used], work=work)
            work["persistent_feature_block_count"] = 1
        else:
            features = torch.empty(state.used, 32, 32, dtype=torch.float32, device="cpu")
            for first in range(0, state.used, 128):
                last = min(first + 128, state.used)
                block = coincidence_features(lags, state.keys[first:last], work=work)
                features[first:last].copy_(block)
                del block
                work["persistent_feature_block_count"] += 1
            work["persistent_feature_assembly_bytes"] = features.numel() * features.element_size()
            work["persistent_feature_copy_values"] = features.numel()
        # The actual completed fixed-cell bank precedes the only trained
        # address layer, and survives only for this call's local teacher.
        scores = (features * self.address_weight).sum(dim=(1, 2))
        _finite(scores, "address scores")
        log_prior = F.log_softmax(scores.to(torch.float64), dim=0)
        _finite(log_prior, "initial log belief")
        work.update({
            "query_raw_bytes": len(query),
            "query_eos_symbols": 1,
            "query_lag_shift_steps": len(query) + 1,
            "query_lag_shift_values": (len(query) + 1) * 31 * 18,
            "query_lag_shift_copy_values": (len(query) + 1) * 31 * 18,
            "query_input_rail_values": (len(query) + 1) * 18,
            "query_token_bytes": (len(query) + 1) * 8,
            "query_raw_lag_bytes": 32 * 18,
            "address_feature_slots": state.used,
            "address_score_synapse_evaluations": state.used * 1024,
            "address_score_reductions": state.used * 1023,
            "address_score_evaluations": state.used,
            "address_normalizations": 1,
        })
        # Retired dense products/reductions and query_population_normalizations
        # stay zero. Post-gate scaling is counted by the actual kernel instead.
        return SequenceQuery(activity, features, scores, log_prior, work)

    @torch.no_grad()
    def predict_with_receipt(
        self, query: bytes, state: TapeState, max_bytes: int
    ) -> CompetitiveSequencePrediction:
        """One query, one competition, and one actually traversed content branch."""
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        event = self._query(query, state)
        competition = compete512(event.address_scores)
        diagnostic_log_prior = event.log_prior.clone()
        work = dict(event.work)
        work.update(competition.work)
        # Free recall does not need the forward X bank after address competition.
        del event
        cursors = torch.tensor([competition.winner_index], dtype=torch.int64, device="cpu")
        self.memory._check_cursors(state, cursors)
        work["branch_cursor_initializations"] = 1
        # This bank is deliberately still evaluated over ALL occupied slots.
        bank_cursors = torch.arange(state.used, dtype=torch.int64, device="cpu")
        work["codec_bank_slot_indices"] = state.used
        bank = self._codec_bank(state, bank_cursors, work)
        del bank_cursors
        initial_log_belief = torch.zeros(1, dtype=torch.float64, device="cpu")
        log_belief = initial_log_belief
        output, steps = bytearray(), []
        for position in range(max_bytes + 1):
            self.memory._check_cursors(state, cursors)
            conditional = bank._gather(cursors)
            work["codec_response_gathers"] += 1
            work["codec_response_values_gathered"] += 257 * 4
            work["codec_response_bytes_gathered"] += 257 * (4 + 3 * 8)
            mixture = self._mixture(conditional, log_belief)
            token = int(mixture.argmax())
            receipt = self._observe(position, token, cursors, log_belief, conditional, mixture)
            steps.append(receipt)
            log_belief = receipt.posterior_log_belief
            self._count_step(work, 1)
            if token == BYTE_EOS or position == max_bytes:
                return CompetitiveSequencePrediction(
                    ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
                    tuple(steps), work, initial_log_belief.clone(), log_belief.clone(),
                    competition, diagnostic_log_prior,
                )
            output.append(token)
            cursors = self.memory._advance_checked(state, cursors)
            work["branch_cursor_advances"] += 1
        raise AssertionError("bounded generation must have a stopping decision")

    def accounting(self) -> dict:
        accounting = super().accounting()
        n, lag_pairs, blocks = 512, 512 * 1024, 4
        pairs = n * (n - 1) // 2
        # The inherited memory dispatch already supplies the actual512-slot tape.
        accounting.update({
            "allocated_address_competition_neurons": pairs + n,
            "maximum_address_comparator_neurons": pairs,
            "maximum_address_winner_neurons": n,
            "maximum_competition_signed_synapse_applications": 4 * pairs,
            "maximum_competition_receipt_tensor_bytes": 13 * n + 9 * pairs,
            "maximum_competition_named_allocated_tensor_bytes": 45 * n + 49 * pairs + 16,
            "maximum_free_diagnostic_log_prior_bytes": n * 8,
            "maximum_free_overlapping_codec_response_bytes": (n + 2) * 257 * 28,
            "maximum_saved_codec_probabilities_bytes": n * 257 * 8,
            "maximum_conditional_step_cache_bytes": {
                "sensory_float32": n * 257 * 4,
                "logits_float64": n * 257 * 8,
                "log_probabilities_float64": n * 257 * 8,
                "probabilities_float64": n * 257 * 8,
                "total": n * 257 * 28,
            },
            "maximum_held_codec_response_bank_bytes": n * 257 * 28,
            "maximum_codec_response_gather_bytes": n * 257 * 28,
            "maximum_overlapping_codec_response_bytes": 3 * n * 257 * 28,
            "maximum_log_belief_bytes": n * 8,
            "maximum_current_codec_error_bytes": n * 257 * 8,
            "maximum_scaled_codec_column_error_bytes": n * 257 * 8,
            "maximum_responsibility_bytes": n * 8,
            "maximum_address_feature_bank_float32_bytes": lag_pairs * 4,
            "maximum_address_score_products_float32_bytes": lag_pairs * 4,
            "maximum_address_scores_float32_bytes": n * 4,
            "maximum_address_feature_cast_float64_bytes": lag_pairs * 8,
            "maximum_address_local_products_float64_bytes": lag_pairs * 8,
            "maximum_address_output_error_float64_bytes": n * 8,
            "maximum_coincidence_threshold_evaluations": 19 * lag_pairs,
            "maximum_coincidence_fixed_input_applications": 54 * lag_pairs,
            # These cumulative named allocations include repeated query-side work.
            # Inherited individual kernel-buffer maxima describe one128-slot block.
            "maximum_coincidence_named_allocated_bytes": (
                60 * lag_pairs + 612 * (n + blocks) + 8 * n),
            "maximum_persistent_feature_block_count": blocks,
            "maximum_persistent_feature_assembly_bytes": lag_pairs * 4,
            "maximum_persistent_feature_copy_values": lag_pairs,
            "coincidence_buffer_scope": (
                "individual maximum_coincidence_*_bytes except named_allocated are "
                "per128-slot call; fourcalls at512, query checks/counts repeat; "
                "named allocation sum excludes assembly counted separately; no RSS claim"
            ),
            "address_clock": (
                "evaluate everyoccupiedslot in128-slot gateblocks, copyallX inoriginalorder; "
                "one full score/teacher reduction, no filtering or feature replay"
            ),
        })
        return accounting
