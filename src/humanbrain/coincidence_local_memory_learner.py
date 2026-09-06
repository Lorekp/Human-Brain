"""Categorical forward coincidence features with existing local output learning.

The fixed encoder and threshold cells precede the shared temporal address
synapses in both free and corrective calls. No error returns through those
cells. Competitive single-cursor recall and all-address teaching are inherited.
"""

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_EOS
from humanbrain.coincidence_sequence_memory import CoincidenceSequenceMemory
from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
from humanbrain.local_memory_learner import SequenceQuery, _finite
from humanbrain.neuronal_byte_coincidence import WORK_KEYS as COINCIDENCE_WORK_KEYS
from humanbrain.neuronal_byte_coincidence import coincidence_features, encode_symbol_rails
from humanbrain.neuronal_sequence_memory import TapeState


class CoincidenceLocalMemoryLearner(CompetitiveLocalMemoryLearner):
    """Fresh binary tape, unchanged slow tensor geometry and direct local teacher.

    The old random input_codes buffer remains the action heads' fixed input
    representation. It is not used to retrieve addresses in this reader.
    Inherited strict loading requires this class's new metadata; numerical prior
    transfer with changed forward meaning belongs to an explicit experiment adapter.
    """

    WORK_KEYS = CompetitiveLocalMemoryLearner.WORK_KEYS + COINCIDENCE_WORK_KEYS + (
        "query_lag_shift_values", "query_input_rail_values", "query_token_bytes",
        "query_raw_lag_bytes", "query_lag_shift_copy_values",
    )

    def __init__(self, *, mode: str = "learned", seed: int = 0) -> None:
        # Construct the original slow parameters once. No new persistent buffer
        # is required, so the base's dynamic parameter check remains complete.
        super().__init__(mode=mode, seed=seed)
        self.memory = CoincidenceSequenceMemory()

    def get_extra_state(self) -> dict:
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "local-neuronal-coincidence-address-learner-v1",
            "query_size": 576,
            "input_code": (
                "unchanged fixed normal258x32/private seed+2 for action heads only; "
                "address input uses stateless nine-bit low-first (zero,one) Boolean rails"
            ),
            "address_rail_size": 18,
            "address": (
                "actual uint8 rail additions then >=2 AND cells; uint8 sum of18 "
                "AND activities then >=9 match cells at every query/key lag pair; "
                "normalize completed match bank before score_s=sum_ij B[i,j]*X_s[i,j]"
            ),
            "address_initialization": (
                "unchanged32x32 temporal Xavier values/private seed+4; changed categorical "
                "input meaning, not functional parity or1024-input fan-Xavier"
            ),
            "address_activity": (
                "actual batched float32[N,32,32] completed coincidence bank; held through "
                "current teacher prefix; gate scratch released before local accumulation"
            ),
            "query": (
                "raw Boolean32x18 newest-first lags including EOS256; reset each query; "
                "empty allzero; bytes0..255; reserved BOS257 never injected"
            ),
            "query_normalization": (
                "no normalization before gates; X=M*float32(sqrt32)/sqrt(float32(Lq*Lk)) "
                "with denominator clamped to1; empty key M=0; L counts occupied lag rows"
            ),
            "memory": (
                "fresh bool[128,32,18] preceding raw-byte rails; no keyEOS/BOS; "
                "local unit-rate binary endpoint copy; unchanged sensory and adjacency"
            ),
            "query_eos_feature_row": (
                "zero against raw-only keys; stored B row0 retained but inactive"
            ),
            "upstream_dtype": "torch.bool rails; torch.uint8 gate currents; torch.float32 X",
            "retention": (
                "typed binary literal keys and unchanged sensory/layout; no conversion "
                "of old normalized random-code tapes or semantic compression claim"
            ),
        })
        return metadata

    @torch.no_grad()
    def ingest(self, documents: Sequence[bytes], state: TapeState | None = None) -> TapeState:
        """Append raw document associations without a teacher or address selector."""
        self._check_parameters()
        if state is not None:
            self._check_state(state, populated=False)
        result = self.memory.ingest(
            documents, state, writes_enabled=self.mode != "writes_disabled"
        )
        self._check_state(result, populated=False)
        return result

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
        features = coincidence_features(lags, state.keys[:state.used], work=work)
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

    def accounting(self) -> dict:
        accounting = super().accounting()
        n, pairs = 128, 128 * 32 * 32
        accounting.update({
            "query_lag_neurons": 576,
            "normalized_query_neurons": 0,
            "query_lag_forward_connections": 31 * 18,
            "query_population_squared_energy": 32 * 9,
            "query_population_squared_energy_scope": (
                "maximum288; actual9*min(raw query bytes+1 EOS,32)"
            ),
            "address_feature_squared_energy_bound": 32,
            "address_feature_products_per_slot": 0,
            "address_feature_channel_reductions_per_slot": 0,
            "address_feature_and_cells_per_slot": 18 * 1024,
            "address_feature_match_cells_per_slot": 1024,
            "address_feature_fixed_input_applications_per_slot": 54 * 1024,
            "address_raw_key_active_temporal_synapses": 31 * 32,
            "fixed_address_codebook_bytes": 0,
            "query_lag_bytes": 576,
            "normalized_query_activity_bytes": 0,
            "query_raw_activity_aliases_lags": True,
            "query_lag_shift_copy_bytes": 31 * 18,
            "query_input_token_bytes_per_symbol": 8,
            "query_encoded_rail_bytes_per_symbol": 18,
            "query_encoder_named_bytes_per_symbol": 108,
            "query_encoder_bit_position_bytes": 72,
            "maximum_coincidence_threshold_evaluations": 19 * pairs,
            "maximum_coincidence_fixed_input_applications": 54 * pairs,
            "maximum_coincidence_operand_cast_bytes": 576 * (n + 1),
            "maximum_coincidence_rail_current_bytes": 18 * pairs,
            "maximum_coincidence_and_activity_bytes": 18 * pairs,
            "maximum_coincidence_match_input_bytes": 18 * pairs,
            "maximum_coincidence_match_current_bytes": pairs,
            "maximum_coincidence_match_activity_bytes": pairs,
            "maximum_coincidence_occupancy_bytes": 32 * (n + 1),
            "maximum_coincidence_occupied_count_bytes": 4 * (n + 1),
            "maximum_coincidence_denominator_bytes": 4 * n,
            "maximum_coincidence_gain_bytes": 4 * n,
            "maximum_coincidence_named_allocated_bytes": 60 * pairs + 612 * (n + 1) + 8 * n,
            "address_clock": (
                "one shared18-AND/one-match fixed circuit time-multiplexed over all "
                "occupied slots and1024 lag pairs; actual batched currents and activities; "
                "occupied-population scaling before shared1024-synapse score; retain X "
                "through current all-address teacher then discard; no backward gate error"
            ),
            "coincidence_scratch_clock": (
                "uint8 current overlaps Boolean AND; release current/operands before "
                "AND-to-uint8 cast; release AND before match sum; release match input "
                "before threshold; release match current before float32 feature cast; "
                "release match activity before in-place feature scaling; all gate scratch "
                "gone before inherited teacher float64 X cast/local products"
            ),
            "work_scope": (
                "query kernel counters include its encoder and rail validation; ordinary "
                "parameter/state checks are boundary validation outside those counters; "
                "named allocation bytes are cumulative explicit tensors, not RSS bounds"
            ),
        })
        return accounting
