"""Bounded competitive free receipts with unchanged marginal/teacher formats.

Pure stdlib and the frozen teacher-free v1 helpers only. The caller supplies
expected_policy from trusted arm and phase metadata. These writer-side consistency checks
do not reconstruct learned scores, codec responses or parameter updates.
"""

from __future__ import annotations

import json
import math
import struct

from experiments.r4_integrated import neuronal_splice_receipts_v1 as v1

LINE_LIMIT = TREE_LIMIT = 2 << 20
READER_SCHEMA = "neuronal-splice-competitive-reader-v2"
COMPETITION_SCHEMA = "neuronal-address-competition-v1"


def bounded_record(row):
    v1.require(type(row) is dict, "record must be an object")
    v1.require(v1.tree_bytes(row) <= TREE_LIMIT, "receipt exceeds tree limit")
    encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("ascii")
    v1.require(len(encoded) <= LINE_LIMIT, "receipt exceeds line limit")
    return encoded


def _policy(expected_policy, *, free_only=False):
    policies = ("competitive_free", "marginal_free", "all_address_teacher")
    v1.require(type(expected_policy) is str and expected_policy in policies,
               "expected explicit reader policy and phase")
    v1.require(not free_only or expected_policy != "all_address_teacher",
               "prediction requires a free reader policy")


def _adjacency(successor):
    v1.require(type(successor) is list and 1 <= len(successor) <= 128,
               "invalid reader population")
    for index, target in enumerate(successor):
        v1.require(type(target) is int and target in (index, index + 1)
                   and target < len(successor), "invalid forward tape adjacency")


def competition_work(used):
    v1.require(type(used) is int and 1 <= used <= 128, "invalid competition population")
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


def competitive_work(used, decisions, query_raw_bytes):
    """Actual query, full occupied codec bank, and one gathered branch per step."""
    v1.require(type(decisions) is int and 1 <= decisions <= 17, "invalid decision count")
    v1.require(type(query_raw_bytes) is int and 0 <= query_raw_bytes <= 10**12,
               "invalid query byte count")
    work = competition_work(used)
    work.update({
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
        "codec_bank_slot_indices": used,
        "codec_response_gathers": decisions,
        "codec_response_values_gathered": decisions * 1028,
        "codec_response_bytes_gathered": decisions * 7196,
        "mixture_output_reductions": decisions * 257,
        "prefix_belief_updates": decisions,
        "branch_cursor_initializations": 1,
        "branch_cursor_advances": decisions - 1,
        "teacher_offers": 0,
        "codec_update_group_offers": 0,
        "address_update_group_offers": 0,
        "enabled_codec_update_groups": 0,
        "enabled_address_update_groups": 0,
    })
    return work


def validate_competition(trace, *, used):
    """Check every saved threshold current and bit; never call the neural circuit."""
    expected_work = competition_work(used)
    v1.exact_keys(trace, ("schema", "scores", "pair_currents", "comparator_activity",
                          "winner_currents", "winner_activity", "winner_index", "work"))
    v1.require(trace["schema"] == COMPETITION_SCHEMA, "competition schema mismatch")
    scores = v1.vector(trace["scores"], used)
    for score in scores:
        try:
            restored = struct.unpack("!f", struct.pack("!f", score))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError("competition score is not finite float32") from error
        v1.require(restored == score, "competition score is not exact float32")
    pairs = used * (used - 1) // 2
    currents = v1.vector(trace["pair_currents"], pairs)
    activity = trace["comparator_activity"]
    v1.require(type(activity) is list and len(activity) == pairs
               and all(type(value) is bool for value in activity), "invalid comparator activity")
    winner_currents = [index - (used - 1) for index in range(used)]
    offset = 0
    for left in range(used):
        for right in range(left + 1, used):
            expected = scores[left] - scores[right]
            v1.require(currents[offset] == expected, "pair current mismatch")
            bit = expected >= 0
            v1.require(activity[offset] is bit, "comparator threshold mismatch")
            winner_currents[left] += int(bit)
            winner_currents[right] -= int(bit)
            offset += 1
    v1.require(v1.vector(trace["winner_currents"], used) == winner_currents,
               "winner current mismatch")
    winners = trace["winner_activity"]
    v1.require(type(winners) is list and len(winners) == used
               and all(type(value) is bool for value in winners), "invalid winner activity")
    v1.require(winners == [current >= 0 for current in winner_currents]
               and sum(winners) == 1, "winner threshold mismatch")
    index = trace["winner_index"]
    v1.require(type(index) is int and index == sum(i * bit for i, bit in enumerate(winners)),
               "identity readout mismatch")
    v1.validate_work(trace["work"])
    v1.require(trace["work"] == expected_work, "competition work mismatch")
    return index


def compact_competition(receipt, *, used):
    row = {"schema": COMPETITION_SCHEMA}
    for name in ("scores", "pair_currents", "comparator_activity", "winner_currents",
                 "winner_activity", "winner_index", "work"):
        row[name] = v1.plain(getattr(receipt, name))
    validate_competition(row, used=used)
    return row


def validate_reader(row, successor, *, expected_policy, max_bytes=16, targets=None):
    _policy(expected_policy)
    _adjacency(successor)
    v1.require(type(row) is dict, "reader receipt must be an object")
    teacher = expected_policy == "all_address_teacher"
    v1.require(row.get("kind") == ("teacher" if teacher else "free"),
               "reader phase differs from expected policy")
    v1.require(teacher or targets is None, "free receipt cannot receive teacher targets")
    if expected_policy != "competitive_free":
        return v1.validate_reader(row, successor, max_bytes=max_bytes, targets=targets)
    v1.exact_keys(row, ("schema", "kind", "initial_log_belief", "steps", "generated_hex",
                        "stop_reason", "work", "competition", "diagnostic_log_prior"))
    v1.require(row["schema"] == READER_SCHEMA and row["kind"] == "free",
               "competitive free schema mismatch")
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
    v1.require(v1.vector(row["initial_log_belief"], 1) == [0.0],
               "competitive initial belief must have one normalized branch")
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
        v1.require(token == max(range(257), key=probabilities.__getitem__),
                   "free token differs from actual mixture argmax")
        v1.require(index == len(steps) - 1 or token != 256, "steps continued after EOS")
        v1.require(v1.vector(step["posterior"], 1) == [0.0],
                   "competitive posterior must have one normalized branch")
        tokens.append(token)
        loss += nll
        if index + 1 < len(steps):
            cursor = successor[cursor]
    eos = tokens[-1] == 256
    v1.require(row["stop_reason"] == ("eos" if eos else "limit"), "stopping receipt mismatch")
    v1.require(eos or len(steps) == max_bytes + 1, "early non-EOS stopping")
    v1.require(bytes(tokens[:-1]) == data, "free emitted bytes mismatch")
    v1.validate_work(row["work"])
    expected_work = competitive_work(used, len(steps), row["work"].get("query_raw_bytes"))
    v1.require(row["work"] == expected_work, "competitive reader work mismatch")
    return {"decisions": len(steps), "loss": loss, "final_log_belief": [0.0],
            "final_cursors": [cursor]}


def compact_reader(receipt, state, *, expected_policy):
    _policy(expected_policy)
    generation = getattr(receipt, "generation", None)
    teacher = expected_policy == "all_address_teacher"
    v1.require((generation is None) == teacher, "actual reader phase differs from expected policy")
    has_competition = hasattr(receipt, "competition") or hasattr(receipt, "diagnostic_log_prior")
    if expected_policy != "competitive_free":
        v1.require(not has_competition, "competitive receipt cannot use inherited format")
        return v1.compact_reader(receipt, state)
    v1.require(hasattr(receipt, "competition") and hasattr(receipt, "diagnostic_log_prior"),
               "competitive receipt requires actual competition and diagnostic activities")
    successor = v1.tape_adjacency(state)
    competition = compact_competition(receipt.competition, used=len(successor))
    belief = v1.plain(receipt.initial_log_belief)
    initial, cursors, steps = belief, [competition["winner_index"]], []
    for index, step in enumerate(receipt.steps):
        v1.require(step.output_step == index, "reader step order changed")
        v1.require(v1.plain(step.prior_log_belief) == belief, "omitted prior is not redundant")
        v1.require(v1.plain(step.branch_cursors) == cursors, "omitted cursor is not deterministic")
        belief = v1.plain(step.posterior_log_belief)
        steps.append({"token": step.observed_token, "nll": step.nll,
                      "mixture": v1.plain(step.mixture_probabilities), "posterior": belief})
        cursors = [successor[cursors[0]]]
    v1.require(v1.plain(receipt.final_log_belief) == belief, "omitted final belief changed")
    row = {"schema": READER_SCHEMA, "kind": "free", "initial_log_belief": initial,
           "steps": steps, "generated_hex": generation.data.hex(),
           "stop_reason": generation.stop_reason, "work": v1.plain(receipt.work),
           "competition": competition, "diagnostic_log_prior": v1.plain(receipt.diagnostic_log_prior)}
    cap = len(steps) - 1 if generation.stop_reason == "limit" else 16
    validate_reader(row, successor, expected_policy=expected_policy, max_bytes=cap)
    return row


def compact_prediction(prediction, state, *, expected_policy):
    _policy(expected_policy, free_only=True)
    row = {"proposal": prediction.proposal.to_record(), "attempt_id": prediction.attempt_id,
           "proposal_sha256": prediction.proposal_sha256, "model_sha256": prediction.model_sha256,
           "memory_sha256": prediction.memory_sha256,
           "choices": {head: v1.compact_head(value, head) for head, value in prediction.choices.items()},
           "payload": compact_reader(prediction.payload, state, expected_policy=expected_policy),
           "work": v1.plain(prediction.work)}
    validate_prediction(row, v1.tape_adjacency(state), expected_policy=expected_policy)
    return row


def validate_prediction(row, successor, *, expected_policy, boundaries=None):
    _policy(expected_policy, free_only=True)
    v1.exact_keys(row, ("proposal", "attempt_id", "proposal_sha256", "model_sha256",
                        "memory_sha256", "choices", "payload", "work"))
    v1.require(type(row["attempt_id"]) is str and 0 < len(row["attempt_id"]) <= 128,
               "invalid attempt identity")
    for name in ("proposal_sha256", "model_sha256", "memory_sha256"):
        v1.hash_value(row[name])
    v1.require(v1.proposal_digest(row["proposal"]) == row["proposal_sha256"], "proposal hash mismatch")
    v1.exact_keys(row["choices"], v1.HEAD_LIMITS)
    for head in v1.HEAD_LIMITS:
        v1.validate_head(row["choices"][head], head)
    proposal = row["proposal"]
    v1.require(proposal["operation"] == ("STOP" if row["choices"]["operation"]["selected_index"] == 0
                                        else "SPLICE"), "operation receipt mismatch")
    validate_reader(row["payload"], successor, expected_policy=expected_policy)
    v1.require(row["payload"]["kind"] == "free", "prediction contains a teacher")
    v1.require(proposal["replacement_hex"] == row["payload"]["generated_hex"]
               and proposal["stop_reason"] == row["payload"]["stop_reason"], "payload mismatch")
    if boundaries is not None:
        v1.require(len(boundaries) == len(row["choices"]["start"]["logits"]), "start population mismatch")
        path, start = boundaries[row["choices"]["start"]["selected_index"]]
        v1.require((proposal["path"], proposal["start"]) == (path, start), "start selection mismatch")
        ends = [offset for name, offset in boundaries if name == path and offset >= start]
        v1.require(len(ends) == len(row["choices"]["end"]["logits"]), "end population mismatch")
        v1.require(proposal["end"] == ends[row["choices"]["end"]["selected_index"]],
                   "end selection mismatch")
    v1.exact_keys(row["work"], ("action", "reader"))
    for counters in row["work"].values():
        v1.validate_work(counters)
    v1.require(row["work"]["reader"] == row["payload"]["work"], "reader work mismatch")


def compact_feedback(feedback, state):
    """Corrective teaching is the unchanged all-address v1 receipt."""
    reader = feedback.reader_feedback
    return {"attempt_id": feedback.attempt_id, "action_updates": v1.plain(feedback.action_updates),
            "action_losses": v1.plain(feedback.action_losses),
            "teacher_choices": {head: v1.compact_head(value, head)
                                for head, value in feedback.teacher_choices.items()},
            "reader": None if reader is None else compact_reader(
                reader, state, expected_policy="all_address_teacher"),
            "reader_updates": None if reader is None else {
                "codec": reader.codec_updated, "address": reader.address_updated},
            "work": v1.plain(feedback.work)}
