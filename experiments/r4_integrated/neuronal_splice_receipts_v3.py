"""Bounded competitive receipts for explicitly selected address frontends.

Pure stdlib and the frozen teacher-free v1/v2 helpers only. The caller supplies
expected_policy and expected_frontend from trusted arm/phase metadata. Complete
competition traces are retained; fixed coincidence gate arrays, learned scores,
codec responses and parameter updates are not reconstructed by these checks.
"""

from __future__ import annotations

import math

from experiments.r4_integrated import neuronal_splice_receipts_v1 as v1
from experiments.r4_integrated.neuronal_splice_receipts_v2 import (
    bounded_record as bounded_record,
)
from experiments.r4_integrated.neuronal_splice_receipts_v2 import (
    compact_competition,
    competition_work,
    validate_competition,
)

LINE_LIMIT = TREE_LIMIT = 2 << 20
READER_SCHEMA = "neuronal-splice-competitive-reader-v3"
COMPETITION_SCHEMA = "neuronal-address-competition-v1"


def _policy(expected_policy, *, free_only=False):
    policies = ("competitive_free", "all_address_teacher")
    v1.require(
        type(expected_policy) is str and expected_policy in policies,
        "expected explicit reader policy and phase",
    )
    v1.require(
        not free_only or expected_policy != "all_address_teacher",
        "prediction requires a free reader policy",
    )


def _frontend(expected_frontend):
    v1.require(
        type(expected_frontend) is str
        and expected_frontend in ("coincidence", "graded_random_code"),
        "expected explicit address frontend",
    )


def coincidence_work(used, query_raw_bytes):
    """Source-derived actual query encoder/gate work, excluding state validation."""
    symbols, features, rows = query_raw_bytes + 1, used * 1024, 32 * (used + 1)
    return {
        "rail_encode_symbols": symbols,
        "rail_encode_bit_positions": 9,
        "rail_encode_bit_shifts": 9 * symbols,
        "rail_encode_bit_masks": 9 * symbols,
        "rail_encode_boolean_cast_values": 9 * symbols,
        "rail_encode_complements": 9 * symbols,
        "rail_encode_output_values": 18 * symbols,
        "rail_encode_named_allocated_bytes": 108 * symbols + 72,
        "rail_validation_rows": rows,
        "rail_validation_complementary_pair_checks": 9 * rows,
        "rail_validation_special_policy_rows": rows,
        "coincidence_feature_calls": 1,
        "coincidence_slots": used,
        "coincidence_lag_pairs": features,
        "coincidence_and_cell_evaluations": 18 * features,
        "coincidence_match_cell_evaluations": features,
        "coincidence_threshold_evaluations": 19 * features,
        "coincidence_and_input_applications": 36 * features,
        "coincidence_match_input_applications": 18 * features,
        "coincidence_input_applications": 54 * features,
        "coincidence_pair_additions": 18 * features,
        "coincidence_rail_sum_additions": 17 * features,
        "coincidence_operand_cast_values": 576 * (used + 1),
        "coincidence_match_input_cast_values": 18 * features,
        "coincidence_occupancy_rail_reductions": 544 * (used + 1),
        "coincidence_occupancy_count_additions": 31 * (used + 1),
        "coincidence_population_products": used,
        "coincidence_population_square_roots": used,
        "coincidence_population_clamps": used,
        "coincidence_gain_fills": used,
        "coincidence_gain_divisions": used,
        "coincidence_feature_cast_values": features,
        "coincidence_feature_scalings": features,
        "coincidence_named_operand_cast_bytes": 576 * (used + 1),
        "coincidence_named_rail_current_bytes": 18 * features,
        "coincidence_named_and_activity_bytes": 18 * features,
        "coincidence_named_match_input_bytes": 18 * features,
        "coincidence_named_match_current_bytes": features,
        "coincidence_named_match_activity_bytes": features,
        "coincidence_named_feature_bytes": 4 * features,
        "coincidence_named_occupancy_bytes": rows,
        "coincidence_named_occupied_count_bytes": 4 * (used + 1),
        "coincidence_named_denominator_bytes": 4 * used,
        "coincidence_named_gain_bytes": 4 * used,
        "coincidence_named_allocated_bytes": 60 * features + 612 * (used + 1) + 8 * used,
        "query_lag_shift_values": 558 * symbols,
        "query_input_rail_values": 18 * symbols,
        "query_token_bytes": 8 * symbols,
        "query_raw_lag_bytes": 576,
        "query_lag_shift_copy_values": 558 * symbols,
    }


def _adjacency(successor):
    v1.require(type(successor) is list and 1 <= len(successor) <= 128, "invalid reader population")
    for index, target in enumerate(successor):
        v1.require(
            type(target) is int and target in (index, index + 1) and target < len(successor),
            "invalid forward tape adjacency",
        )


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
    """Actual work, including the full teacher bank or committed free gathers."""
    _policy(expected_policy)
    _frontend(expected_frontend)
    v1.require(type(decisions) is int and 1 <= decisions <= 17, "invalid decision count")
    v1.require(
        type(query_raw_bytes) is int and 0 <= query_raw_bytes <= 10**12, "invalid query byte count"
    )
    teacher = expected_policy == "all_address_teacher"
    for flag in (enabled_codec_update_groups, enabled_address_update_groups):
        v1.require(type(flag) is int and flag in (0, 1), "invalid enabled teacher group")
        v1.require(teacher or flag == 0, "free call cannot enable teaching")
    v1.require(type(used) is int and 1 <= used <= 128, "invalid reader population")
    branches = used if teacher else 1
    gathers = used * (decisions - 1) if teacher else decisions
    work = {} if teacher else competition_work(used)
    work.update(
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
            "address_local_synapse_contributions": 0,
            "address_local_slot_reductions": 0,
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
            "teacher_offers": decisions if teacher else 0,
            "codec_update_group_offers": int(teacher),
            "address_update_group_offers": int(teacher),
            "enabled_codec_update_groups": enabled_codec_update_groups,
            "enabled_address_update_groups": enabled_address_update_groups,
        }
    )
    if teacher:
        work["address_local_synapse_contributions"] = used * 1024
        work["address_local_slot_reductions"] = (used - 1) * 1024
    else:
        work["codec_bank_slot_indices"] = used
    if expected_frontend == "coincidence":
        work.update(coincidence_work(used, query_raw_bytes))
        work["query_population_normalizations"] = 0
        work["address_feature_products"] = 0
        work["address_feature_channel_reductions"] = 0
    return work


def _validate_reader_work(
    work, used, decisions, *, expected_policy, expected_frontend, query_raw_bytes
):
    v1.validate_work(work)
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
    v1.require(work == expected, "reader frontend work mismatch")


def validate_reader(
    row,
    successor,
    *,
    expected_policy,
    expected_frontend,
    max_bytes=16,
    targets=None,
    query_raw_bytes=None,
):
    _policy(expected_policy)
    _frontend(expected_frontend)
    _adjacency(successor)
    v1.require(type(row) is dict, "reader receipt must be an object")
    teacher = expected_policy == "all_address_teacher"
    v1.require(
        row.get("kind") == ("teacher" if teacher else "free"),
        "reader phase differs from expected policy",
    )
    v1.require(teacher or targets is None, "free receipt cannot receive teacher targets")
    if teacher:
        summary = v1.validate_reader(row, successor, max_bytes=max_bytes, targets=targets)
        _validate_reader_work(
            row["work"],
            len(successor),
            summary["decisions"],
            expected_policy=expected_policy,
            expected_frontend=expected_frontend,
            query_raw_bytes=query_raw_bytes,
        )
        return summary
    v1.exact_keys(
        row,
        (
            "schema",
            "kind",
            "initial_log_belief",
            "steps",
            "generated_hex",
            "stop_reason",
            "work",
            "competition",
            "diagnostic_log_prior",
        ),
    )
    v1.require(
        row["schema"] == READER_SCHEMA and row["kind"] == "free", "competitive free schema mismatch"
    )
    v1.require(targets is None, "free receipt cannot receive teacher targets")
    v1.require(type(max_bytes) is int and 0 <= max_bytes <= 16, "invalid reader cap")
    used = len(successor)
    cursor = validate_competition(row["competition"], used=used)
    scores = row["competition"]["scores"]
    peak = max(scores)
    centered = [score - peak for score in scores]
    denominator = v1.logsumexp(centered)
    diagnostic = v1.vector(row["diagnostic_log_prior"], used)
    for actual, value in zip(diagnostic, centered, strict=True):
        expected = value - denominator
        v1.close(actual, expected, atol=max(2e-10, 4 * math.ulp(expected)), rtol=0)
    v1.close(v1.logsumexp(diagnostic), 0)
    v1.require(
        v1.vector(row["initial_log_belief"], 1) == [0.0],
        "competitive initial belief must have one normalized branch",
    )
    steps = row["steps"]
    v1.require(type(steps) is list and 1 <= len(steps) <= max_bytes + 1, "invalid step count")
    data = v1.hex_bytes(row["generated_hex"], max_bytes)
    tokens, loss = [], 0.0
    for index, step in enumerate(steps):
        v1.exact_keys(step, ("token", "nll", "mixture", "posterior"))
        token = step["token"]
        v1.require(type(token) is int and 0 <= token <= 256, "invalid observed token")
        probabilities = v1.vector(step["mixture"], 257)
        v1.require(all(0 <= value <= 1 for value in probabilities), "invalid probability")
        v1.close(math.fsum(probabilities), 1)
        nll = v1.number(step["nll"])
        v1.require(nll >= -2e-10, "negative NLL")
        if probabilities[token] > 1e-250:
            v1.close(-math.log(probabilities[token]), nll)
        else:
            v1.close(probabilities[token], math.exp(-nll), atol=5e-324, rtol=2e-10)
        v1.require(
            token == max(range(257), key=probabilities.__getitem__),
            "free token differs from actual mixture argmax",
        )
        v1.require(index == len(steps) - 1 or token != 256, "steps continued after EOS")
        v1.require(
            v1.vector(step["posterior"], 1) == [0.0],
            "competitive posterior must have one normalized branch",
        )
        tokens.append(token)
        loss += nll
        if index + 1 < len(steps):
            cursor = successor[cursor]
    eos = tokens[-1] == 256
    v1.require(row["stop_reason"] == ("eos" if eos else "limit"), "stopping receipt mismatch")
    v1.require(eos or len(steps) == max_bytes + 1, "early non-EOS stopping")
    v1.require(bytes(tokens[:-1]) == data, "free emitted bytes mismatch")
    _validate_reader_work(
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


def compact_reader(receipt, state, *, expected_policy, expected_frontend, query_raw_bytes=None):
    _policy(expected_policy)
    _frontend(expected_frontend)
    generation = getattr(receipt, "generation", None)
    teacher = expected_policy == "all_address_teacher"
    v1.require((generation is None) == teacher, "actual reader phase differs from expected policy")
    has_competition = hasattr(receipt, "competition") or hasattr(receipt, "diagnostic_log_prior")
    if expected_policy != "competitive_free":
        v1.require(not has_competition, "competitive receipt cannot use inherited format")
        row = v1.compact_reader(receipt, state)
        validate_reader(
            row,
            v1.tape_adjacency(state),
            expected_policy=expected_policy,
            expected_frontend=expected_frontend,
            query_raw_bytes=query_raw_bytes,
        )
        return row
    v1.require(
        hasattr(receipt, "competition") and hasattr(receipt, "diagnostic_log_prior"),
        "competitive receipt requires actual competition and diagnostic activities",
    )
    successor = v1.tape_adjacency(state)
    competition = compact_competition(receipt.competition, used=len(successor))
    belief = v1.plain(receipt.initial_log_belief)
    initial, cursors, steps = belief, [competition["winner_index"]], []
    for index, step in enumerate(receipt.steps):
        v1.require(step.output_step == index, "reader step order changed")
        v1.require(v1.plain(step.prior_log_belief) == belief, "omitted prior is not redundant")
        v1.require(v1.plain(step.branch_cursors) == cursors, "omitted cursor is not deterministic")
        belief = v1.plain(step.posterior_log_belief)
        steps.append(
            {
                "token": step.observed_token,
                "nll": step.nll,
                "mixture": v1.plain(step.mixture_probabilities),
                "posterior": belief,
            }
        )
        cursors = [successor[cursors[0]]]
    v1.require(v1.plain(receipt.final_log_belief) == belief, "omitted final belief changed")
    row = {
        "schema": READER_SCHEMA,
        "kind": "free",
        "initial_log_belief": initial,
        "steps": steps,
        "generated_hex": generation.data.hex(),
        "stop_reason": generation.stop_reason,
        "work": v1.plain(receipt.work),
        "competition": competition,
        "diagnostic_log_prior": v1.plain(receipt.diagnostic_log_prior),
    }
    cap = len(steps) - 1 if generation.stop_reason == "limit" else 16
    validate_reader(
        row,
        successor,
        expected_policy=expected_policy,
        max_bytes=cap,
        expected_frontend=expected_frontend,
        query_raw_bytes=query_raw_bytes,
    )
    return row


def compact_prediction(
    prediction, state, *, expected_policy, expected_frontend, query_raw_bytes=None
):
    _policy(expected_policy, free_only=True)
    _frontend(expected_frontend)
    row = {
        "proposal": prediction.proposal.to_record(),
        "attempt_id": prediction.attempt_id,
        "proposal_sha256": prediction.proposal_sha256,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "choices": {
            head: v1.compact_head(value, head) for head, value in prediction.choices.items()
        },
        "payload": compact_reader(
            prediction.payload,
            state,
            expected_policy=expected_policy,
            expected_frontend=expected_frontend,
            query_raw_bytes=query_raw_bytes,
        ),
        "work": v1.plain(prediction.work),
    }
    validate_prediction(
        row,
        v1.tape_adjacency(state),
        expected_policy=expected_policy,
        expected_frontend=expected_frontend,
        query_raw_bytes=query_raw_bytes,
    )
    return row


def validate_prediction(
    row, successor, *, expected_policy, expected_frontend, boundaries=None, query_raw_bytes=None
):
    _policy(expected_policy, free_only=True)
    _frontend(expected_frontend)
    v1.exact_keys(
        row,
        (
            "proposal",
            "attempt_id",
            "proposal_sha256",
            "model_sha256",
            "memory_sha256",
            "choices",
            "payload",
            "work",
        ),
    )
    v1.require(
        type(row["attempt_id"]) is str and 0 < len(row["attempt_id"]) <= 128,
        "invalid attempt identity",
    )
    for name in ("proposal_sha256", "model_sha256", "memory_sha256"):
        v1.hash_value(row[name])
    v1.require(
        v1.proposal_digest(row["proposal"]) == row["proposal_sha256"], "proposal hash mismatch"
    )
    v1.exact_keys(row["choices"], v1.HEAD_LIMITS)
    for head in v1.HEAD_LIMITS:
        v1.validate_head(row["choices"][head], head)
    proposal = row["proposal"]
    v1.require(
        proposal["operation"]
        == ("STOP" if row["choices"]["operation"]["selected_index"] == 0 else "SPLICE"),
        "operation receipt mismatch",
    )
    validate_reader(
        row["payload"],
        successor,
        expected_policy=expected_policy,
        expected_frontend=expected_frontend,
        query_raw_bytes=query_raw_bytes,
    )
    v1.require(row["payload"]["kind"] == "free", "prediction contains a teacher")
    v1.require(
        proposal["replacement_hex"] == row["payload"]["generated_hex"]
        and proposal["stop_reason"] == row["payload"]["stop_reason"],
        "payload mismatch",
    )
    if boundaries is not None:
        v1.require(
            len(boundaries) == len(row["choices"]["start"]["logits"]), "start population mismatch"
        )
        path, start = boundaries[row["choices"]["start"]["selected_index"]]
        v1.require(
            (proposal["path"], proposal["start"]) == (path, start), "start selection mismatch"
        )
        ends = [offset for name, offset in boundaries if name == path and offset >= start]
        v1.require(len(ends) == len(row["choices"]["end"]["logits"]), "end population mismatch")
        v1.require(
            proposal["end"] == ends[row["choices"]["end"]["selected_index"]],
            "end selection mismatch",
        )
    v1.exact_keys(row["work"], ("action", "reader"))
    for counters in row["work"].values():
        v1.validate_work(counters)
    v1.require(row["work"]["reader"] == row["payload"]["work"], "reader work mismatch")


def compact_feedback(feedback, state, *, expected_frontend, query_raw_bytes=None):
    """Corrective teaching is the unchanged all-address v1 receipt."""
    reader = feedback.reader_feedback
    _frontend(expected_frontend)
    return {
        "attempt_id": feedback.attempt_id,
        "action_updates": v1.plain(feedback.action_updates),
        "action_losses": v1.plain(feedback.action_losses),
        "teacher_choices": {
            head: v1.compact_head(value, head) for head, value in feedback.teacher_choices.items()
        },
        "reader": None
        if reader is None
        else compact_reader(
            reader,
            state,
            expected_policy="all_address_teacher",
            expected_frontend=expected_frontend,
            query_raw_bytes=query_raw_bytes,
        ),
        "reader_updates": None
        if reader is None
        else {"codec": reader.codec_updated, "address": reader.address_updated},
        "work": v1.plain(feedback.work),
    }
