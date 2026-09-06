"""Independent saved reader checks for categorical and graded address inputs.

No production competitor, new receipt writer, tensor library or model is imported.
These checks establish saved consistency, not unsaved query/codec evaluation or
parameter-update replay. The caller authenticates adjacency and the expected
policy, frontend and query length. Enabled teacher groups need enclosing arm
bindings. Gate work is source-derived; unsaved gate activities are not replayed.
"""

from __future__ import annotations

import math

from experiments.r4_integrated.check_neuronal_address_competition_v1 import (
    _competition_work,
    check_competition_trace,
)
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

READER_SCHEMA = "neuronal-splice-competitive-reader-v3"
COMPETITION_SCHEMA = "neuronal-address-competition-v1"
READER_FIELDS = (
    "kind",
    "initial_log_belief",
    "steps",
    "generated_hex",
    "stop_reason",
    "work",
)
BASE_WORK_KEYS = (
    "query_raw_bytes",
    "query_eos_symbols",
    "query_lag_shift_steps",
    "query_population_normalizations",
    "address_feature_slots",
    "address_feature_products",
    "address_feature_channel_reductions",
    "address_score_synapse_evaluations",
    "address_score_reductions",
    "address_score_evaluations",
    "address_local_synapse_contributions",
    "address_local_slot_reductions",
    "address_normalizations",
    "codec_slot_evaluations",
    "codec_output_potentials",
    "codec_normalizations",
    "codec_response_gathers",
    "codec_response_values_gathered",
    "codec_response_bytes_gathered",
    "mixture_output_reductions",
    "prefix_belief_updates",
    "branch_cursor_initializations",
    "branch_cursor_advances",
    "teacher_offers",
    "codec_update_group_offers",
    "address_update_group_offers",
    "enabled_codec_update_groups",
    "enabled_address_update_groups",
)


def _coincidence_query_work(used, raw):
    """Independent scalar count derivation, without the kernel or writer imports."""
    symbols = raw + 1
    populations = used + 1
    lag_rows = 32 * populations
    matches = 32 * 32 * used
    and_cells = 18 * matches
    work = {
        "rail_encode_symbols": symbols,
        "rail_encode_bit_positions": 9,
        "rail_encode_output_values": 18 * symbols,
        "rail_encode_named_allocated_bytes": 108 * symbols + 72,
        "rail_validation_rows": lag_rows,
        "rail_validation_complementary_pair_checks": 9 * lag_rows,
        "rail_validation_special_policy_rows": lag_rows,
        "coincidence_feature_calls": 1,
        "coincidence_slots": used,
        "coincidence_lag_pairs": matches,
        "coincidence_and_cell_evaluations": and_cells,
        "coincidence_match_cell_evaluations": matches,
        "coincidence_threshold_evaluations": and_cells + matches,
        "coincidence_and_input_applications": 2 * and_cells,
        "coincidence_match_input_applications": and_cells,
        "coincidence_input_applications": 3 * and_cells,
        "coincidence_pair_additions": and_cells,
        "coincidence_rail_sum_additions": 17 * matches,
        "coincidence_operand_cast_values": 18 * lag_rows,
        "coincidence_match_input_cast_values": and_cells,
        "coincidence_occupancy_rail_reductions": 17 * lag_rows,
        "coincidence_occupancy_count_additions": 31 * populations,
        "coincidence_feature_cast_values": matches,
        "coincidence_feature_scalings": matches,
        "coincidence_named_operand_cast_bytes": 18 * lag_rows,
        "coincidence_named_rail_current_bytes": and_cells,
        "coincidence_named_and_activity_bytes": and_cells,
        "coincidence_named_match_input_bytes": and_cells,
        "coincidence_named_match_current_bytes": matches,
        "coincidence_named_match_activity_bytes": matches,
        "coincidence_named_feature_bytes": 4 * matches,
        "coincidence_named_occupancy_bytes": lag_rows,
        "coincidence_named_occupied_count_bytes": 4 * populations,
        "coincidence_named_denominator_bytes": 4 * used,
        "coincidence_named_gain_bytes": 4 * used,
        "coincidence_named_allocated_bytes": 60 * matches + 612 * populations + 8 * used,
        "query_lag_shift_values": 31 * 18 * symbols,
        "query_lag_shift_copy_values": 31 * 18 * symbols,
        "query_input_rail_values": 18 * symbols,
        "query_token_bytes": 8 * symbols,
        "query_raw_lag_bytes": 32 * 18,
    }
    for suffix in ("bit_shifts", "bit_masks", "boolean_cast_values", "complements"):
        work["rail_encode_" + suffix] = 9 * symbols
    for suffix in (
        "population_products",
        "population_square_roots",
        "population_clamps",
        "gain_fills",
        "gain_divisions",
    ):
        work["coincidence_" + suffix] = used
    return work


def expected_reader_work(
    used,
    decisions,
    query_raw_bytes,
    *,
    expected_policy,
    expected_frontend,
    enabled_codec_update_groups=0,
    enabled_address_update_groups=0,
):
    require(type(used) is int and 1 <= used <= 128, "invalid reader population")
    require(type(decisions) is int and 1 <= decisions <= 17, "invalid reader decisions")
    require(
        type(query_raw_bytes) is int and 0 <= query_raw_bytes <= 10**12,
        "invalid trusted query byte count",
    )
    require(expected_policy in ("competitive_free", "all_address_teacher"), "invalid reader policy")
    require(expected_frontend in ("coincidence", "graded_random_code"), "invalid address frontend")
    teacher = expected_policy == "all_address_teacher"
    competitive = not teacher
    for flag in (enabled_codec_update_groups, enabled_address_update_groups):
        require(type(flag) is int and flag in (0, 1), "invalid enabled teacher group")
        require(teacher or flag == 0, "free receipt enables teaching")
    expected = dict.fromkeys(BASE_WORK_KEYS, 0)
    branches = 1 if competitive else used
    gathers = decisions if competitive else used * (decisions - 1)
    expected.update(
        {
            "query_raw_bytes": query_raw_bytes,
            "query_eos_symbols": 1,
            "query_lag_shift_steps": query_raw_bytes + 1,
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
        }
    )
    if teacher:
        expected.update(
            {
                "teacher_offers": decisions,
                "address_local_synapse_contributions": used * 1024,
                "address_local_slot_reductions": (used - 1) * 1024,
                "codec_update_group_offers": 1,
                "address_update_group_offers": 1,
            }
        )
        expected["enabled_codec_update_groups"] = enabled_codec_update_groups
        expected["enabled_address_update_groups"] = enabled_address_update_groups
    if competitive:
        expected.update(_competition_work(used))
        expected["codec_bank_slot_indices"] = used
    if expected_frontend == "coincidence":
        expected.update(_coincidence_query_work(used, query_raw_bytes))
        for key in (
            "query_population_normalizations",
            "address_feature_products",
            "address_feature_channel_reductions",
        ):
            expected[key] = 0
    return expected


def _check_work(work, used, decisions, *, expected_policy, expected_frontend, query_raw_bytes):
    validate_work(work)
    raw = work.get("query_raw_bytes") if query_raw_bytes is None else query_raw_bytes
    expected = expected_reader_work(
        used,
        decisions,
        raw,
        expected_policy=expected_policy,
        expected_frontend=expected_frontend,
        enabled_codec_update_groups=work.get("enabled_codec_update_groups"),
        enabled_address_update_groups=work.get("enabled_address_update_groups"),
    )
    require(work == expected, "reader frontend work mismatch")


def check_reader(
    row,
    successor,
    expected_policy,
    *,
    expected_frontend,
    max_bytes=16,
    targets=None,
    query_raw_bytes=None,
):
    """Validate an explicitly selected policy and return the final cursor summary.

    The unchanged v1 validator handles full teaching.
    Competitive fields are checked independently here. Omitted actual cursor/
    prior equality is a writer check; this checker derives the saved path from
    authenticated adjacency and the complete competition trace.
    """
    require(
        expected_policy in ("competitive_free", "all_address_teacher"),
        "unknown expected reader policy",
    )
    require(expected_frontend in ("coincidence", "graded_random_code"), "unknown address frontend")
    require(type(row) is dict and row.get("kind") in ("free", "teacher"), "invalid reader kind")
    teacher = expected_policy == "all_address_teacher"
    require(row["kind"] == ("teacher" if teacher else "free"), "reader phase mismatch")
    require(teacher or targets is None, "free recall cannot accept correction targets")
    require(type(max_bytes) is int and 0 <= max_bytes <= 16, "invalid reader cap")
    require(type(successor) is list and 1 <= len(successor) <= 128, "invalid tape population")
    used = len(successor)
    for index, target in enumerate(successor):
        require(
            type(target) is int and target in (index, index + 1) and target < used,
            "invalid forward tape adjacency",
        )
    if teacher:
        summary = validate_reader(row, successor, max_bytes=max_bytes, targets=targets)
        _check_work(
            row["work"],
            used,
            summary["decisions"],
            expected_policy=expected_policy,
            expected_frontend=expected_frontend,
            query_raw_bytes=query_raw_bytes,
        )
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
        require(
            math.isclose(actual, expected, rel_tol=0, abs_tol=max(2e-10, 4 * math.ulp(expected))),
            "diagnostic log prior differs from centered score normalization",
        )
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
    _check_work(
        row["work"],
        used,
        len(steps),
        expected_policy=expected_policy,
        expected_frontend=expected_frontend,
        query_raw_bytes=query_raw_bytes,
    )
    return {
        "decisions": len(steps),
        "loss": loss,
        "final_log_belief": [0.0],
        "final_cursors": [cursor],
    }
