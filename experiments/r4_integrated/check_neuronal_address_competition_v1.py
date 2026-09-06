"""Independent scalar checks of saved competition and committed free recall.

No production competitor, new receipt writer, tensor library or model is imported.
These checks establish saved consistency, not unsaved query/codec evaluation or
parameter-update replay. The caller authenticates adjacency and the expected
policy; query length and enabled teacher groups need enclosing evidence bindings.
"""

from __future__ import annotations

import math
import struct

from experiments.r4_integrated.neuronal_splice_receipts_v1 import (
    close,
    exact_keys,
    hex_bytes,
    logsumexp,
    number,
    require,
    validate_reader,
    validate_work,
    vector,
)

READER_SCHEMA = "neuronal-splice-competitive-reader-v2"
COMPETITION_SCHEMA = "neuronal-address-competition-v1"
READER_FIELDS = (
    "kind", "initial_log_belief", "steps", "generated_hex", "stop_reason", "work",
)
BASE_WORK_KEYS = (
    "query_raw_bytes", "query_eos_symbols", "query_lag_shift_steps",
    "query_population_normalizations", "address_feature_slots",
    "address_feature_products", "address_feature_channel_reductions",
    "address_score_synapse_evaluations", "address_score_reductions",
    "address_score_evaluations", "address_local_synapse_contributions",
    "address_local_slot_reductions", "address_normalizations", "codec_slot_evaluations",
    "codec_output_potentials", "codec_normalizations", "codec_response_gathers",
    "codec_response_values_gathered", "codec_response_bytes_gathered",
    "mixture_output_reductions", "prefix_belief_updates", "branch_cursor_initializations",
    "branch_cursor_advances", "teacher_offers", "codec_update_group_offers",
    "address_update_group_offers", "enabled_codec_update_groups", "enabled_address_update_groups",
)


def _competition_work(used):
    pairs = used * (used - 1) // 2
    return {
        "competition_candidates": used,
        "competition_pairs": pairs,
        "competition_threshold_layers": 2,
        "competition_comparator_neurons": pairs,
        "competition_winner_neurons": used,
        "competition_comparator_synapse_applications": 2 * pairs,
        "competition_winner_synapse_applications": 2 * pairs,
        "competition_comparator_subtractions": pairs,
        "competition_winner_accumulations": 2 * pairs,
        "competition_winner_bias_values": used,
        "competition_winner_threshold_subtractions": used,
        "competition_threshold_evaluations": pairs + used,
        "competition_identity_synapse_applications": used,
        "competition_identity_reductions": used - 1,
        "competition_winner_count_reductions": used - 1,
        "competition_index_values_constructed": 2 * pairs + used,
        "competition_receipt_tensor_bytes": 13 * used + 9 * pairs,
        "competition_named_allocated_tensor_bytes": 45 * used + 49 * pairs + 16,
        "competition_pair_index_bytes": 16 * pairs,
        "competition_score_gather_bytes": 16 * pairs,
    }


def _bits(values, count):
    require(type(values) is list and len(values) == count, "threshold geometry mismatch")
    require(all(type(value) is bool for value in values), "threshold activities must be booleans")
    return values


def check_competition_trace(trace, used):
    """Check every fixed current/threshold and return its unique identity readout."""
    require(type(used) is int and 1 <= used <= 128, "invalid occupied population")
    exact_keys(trace, (
        "schema", "scores", "pair_currents", "comparator_activity", "winner_currents",
        "winner_activity", "winner_index", "work",
    ))
    require(trace["schema"] == COMPETITION_SCHEMA, "wrong competition schema")
    scores = vector(trace["scores"], used)
    for score in scores:
        try:
            roundtrip = struct.unpack("!f", struct.pack("!f", score))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError("score is outside float32") from error
        require(roundtrip == score, "score is not an exact float32 value")
    pairs = used * (used - 1) // 2
    currents = vector(trace["pair_currents"], pairs)
    activities = _bits(trace["comparator_activity"], pairs)
    # Accumulate the fixed signed wiring independently, in upper-triangle order.
    expected_winners = [index - (used - 1) for index in range(used)]
    pair = 0
    for left in range(used):
        for right in range(left + 1, used):
            current = scores[left] - scores[right]
            require(currents[pair] == current, "pair current mismatch")
            active = current >= 0
            require(activities[pair] is active, "comparator threshold mismatch")
            expected_winners[left] += int(active)
            expected_winners[right] -= int(active)
            pair += 1
    winners = vector(trace["winner_currents"], used)
    require(winners == expected_winners, "winner current mismatch")
    bits = _bits(trace["winner_activity"], used)
    require(bits == [current >= 0 for current in expected_winners], "winner threshold mismatch")
    require(sum(bits) == 1, "competition did not commit exactly one address")
    winner = sum(index * int(active) for index, active in enumerate(bits))
    require(type(trace["winner_index"]) is int and trace["winner_index"] == winner,
            "identity readout mismatch")
    validate_work(trace["work"])
    require(trace["work"] == _competition_work(used), "competition work mismatch")
    return winner


def _check_work(work, used, decisions, *, teacher, competitive):
    validate_work(work)
    require("query_raw_bytes" in work, "missing query length counter")
    expected = dict.fromkeys(BASE_WORK_KEYS, 0)
    branches = 1 if competitive else used
    gathers = decisions if competitive else used * (decisions - 1)
    expected.update({
        "query_raw_bytes": work["query_raw_bytes"],
        "query_eos_symbols": 1,
        "query_lag_shift_steps": work["query_raw_bytes"] + 1,
        "query_population_normalizations": 1,
        "address_feature_slots": used,
        "address_feature_products": used * 32768,
        "address_feature_channel_reductions": used * 31744,
        "address_score_synapse_evaluations": used * 1024,
        "address_score_reductions": used * 1023,
        "address_score_evaluations": used,
        "address_normalizations": 1,
        "codec_slot_evaluations": used,
        "codec_output_potentials": used * 257,
        "codec_normalizations": used,
        "codec_response_gathers": gathers,
        "codec_response_values_gathered": gathers * 1028,
        "codec_response_bytes_gathered": gathers * 7196,
        "mixture_output_reductions": decisions * 257,
        "prefix_belief_updates": branches * decisions,
        "branch_cursor_initializations": branches,
        "branch_cursor_advances": branches * (decisions - 1),
    })
    if teacher:
        expected.update({
            "teacher_offers": decisions,
            "address_local_synapse_contributions": used * 1024,
            "address_local_slot_reductions": (used - 1) * 1024,
            "codec_update_group_offers": 1,
            "address_update_group_offers": 1,
        })
        for key in ("enabled_codec_update_groups", "enabled_address_update_groups"):
            require(key in work and work[key] in (0, 1), "invalid enabled teacher group")
            expected[key] = work[key]
    if competitive:
        expected.update(_competition_work(used))
        expected["codec_bank_slot_indices"] = used
    require(work == expected, "reader work mismatch")


def check_reader(row, successor, expected_policy, max_bytes=16, targets=None):
    """Validate an explicitly selected policy and return the final cursor summary.

    The unchanged v1 validator handles marginal free recall and full teaching.
    Competitive fields are checked independently here. Omitted actual cursor/
    prior equality is a writer check; this checker derives the saved path from
    authenticated adjacency and the complete competition trace.
    """
    require(expected_policy in ("competitive_free", "marginal_free", "all_address_teacher"),
            "unknown expected reader policy")
    require(type(row) is dict and row.get("kind") in ("free", "teacher"), "invalid reader kind")
    teacher = expected_policy == "all_address_teacher"
    require(row["kind"] == ("teacher" if teacher else "free"), "reader phase mismatch")
    require(teacher or targets is None, "free recall cannot accept correction targets")
    require(type(max_bytes) is int and 0 <= max_bytes <= 16, "invalid reader cap")
    require(type(successor) is list and 1 <= len(successor) <= 128, "invalid tape population")
    used = len(successor)
    for index, target in enumerate(successor):
        require(type(target) is int and target in (index, index + 1) and target < used,
                "invalid forward tape adjacency")
    if teacher or expected_policy == "marginal_free":
        summary = validate_reader(row, successor, max_bytes=max_bytes, targets=targets)
        _check_work(row["work"], used, summary["decisions"], teacher=teacher, competitive=False)
        return summary

    exact_keys(row, (*READER_FIELDS, "schema", "competition", "diagnostic_log_prior"))
    require(row["schema"] == READER_SCHEMA, "wrong competitive reader schema")
    cursor = check_competition_trace(row["competition"], used)
    scores = vector(row["competition"]["scores"], used)
    diagnostic = vector(row["diagnostic_log_prior"], used)
    peak = max(scores)
    centered = [score - peak for score in scores]
    normalizer = math.log(math.fsum(math.exp(score) for score in centered))
    for actual, score in zip(diagnostic, centered, strict=True):
        expected = score - normalizer
        require(math.isclose(actual, expected, rel_tol=0,
                             abs_tol=max(2e-10, 4 * math.ulp(expected))),
                "diagnostic log prior differs from centered score normalization")
    close(logsumexp(diagnostic), 0)
    require(vector(row["initial_log_belief"], 1) == [0.0], "free initial belief is not committed")
    steps = row["steps"]
    require(type(steps) is list and 1 <= len(steps) <= max_bytes + 1, "invalid step count")
    data = hex_bytes(row["generated_hex"], max_bytes)
    tokens, loss = [], 0.0
    for index, step in enumerate(steps):
        exact_keys(step, ("token", "nll", "mixture", "posterior"))
        token = step["token"]
        require(type(token) is int and 0 <= token <= 256, "invalid free token")
        probabilities = vector(step["mixture"], 257)
        require(all(0 <= value <= 1 for value in probabilities), "invalid probability")
        close(math.fsum(probabilities), 1)
        require(token == max(range(257), key=probabilities.__getitem__), "free maximum mismatch")
        nll = number(step["nll"])
        require(nll >= -2e-10, "negative free NLL")
        # A maximum among 257 normalized probabilities cannot underflow.
        require(probabilities[token] > 0, "free maximum has zero probability")
        close(nll, -math.log(probabilities[token]))
        require(vector(step["posterior"], 1) == [0.0], "free posterior is not committed")
        require(index == len(steps) - 1 or token != 256, "free recall continued after EOS")
        tokens.append(token)
        loss += nll
        if index + 1 < len(steps):
            cursor = successor[cursor]
    eos = tokens[-1] == 256
    require(row["stop_reason"] == ("eos" if eos else "limit"), "free stopping mismatch")
    require(eos or len(steps) == max_bytes + 1, "early non-EOS stopping")
    require(bytes(tokens[:-1]) == data, "free byte assembly mismatch")
    _check_work(row["work"], used, len(steps), teacher=False, competitive=True)
    return {"decisions": len(steps), "loss": loss, "final_log_belief": [0.0],
            "final_cursors": [cursor]}
