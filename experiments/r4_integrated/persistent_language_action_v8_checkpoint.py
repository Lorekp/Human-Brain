"""V8 source semantics over the unchanged idle-boundary durability protocol.

Every observed source advances the public schedule and simple history. Only its
frozen admission flag permits neuronal tape growth. No payload or source check
is bypassed, and no unresolved interaction is checkpointed.
"""

from __future__ import annotations

import hashlib
import time

from experiments.r4_integrated.persistent_session_checkpoint import (
    NEURAL_ROUTES,
    ROUTES,
    PersistentSessionCheckpoint,
    _baseline_values,
    _integer,
    _keys,
    _raw,
    _restore_baseline,
    _sha,
)

SCHEMA = "persistent-language-action-stream-v8"
INITIAL_FACTS = 1614
SOURCE_LIMIT = 3484
SOURCE_SYMBOL_LIMIT = 10500
DOCUMENT_KEYS = ("document_id", "text_hex", "sha256", "neural_tape_admission", "source_kind")
HASH_FIELDS = (
    "model_before_sha256", "model_after_sha256",
    "memory_before_sha256", "memory_after_sha256",
    "controller_rng_before_sha256", "controller_rng_after_sha256",
    "controller_slow_before_sha256", "controller_slow_after_sha256",
)


def controller_hashes(reader, work=None):
    """Hash actual published controller state; never infer it from journal rows."""
    from humanbrain.neuronal_splice_learner import _digest

    state = reader.controller.state_dict()
    result = {
        "rng_sha256": _digest({"rng_state": state["rng_state"]}, work),
        "slow_sha256": _digest({key: value for key, value in state.items()
                                 if key != "rng_state"}, work),
    }
    if work is not None:
        work["checkpoint_v8_controller_hash_calls"] = (
            work.get("checkpoint_v8_controller_hash_calls", 0) + 1)
    return result


class PersistentLanguageActionV8Checkpoint(PersistentSessionCheckpoint):
    """Same HEAD/intents/leases; additional frozen-source and actual RNG checks.

    A small tuple of scalar hashes is anchored to the last successful boundary.
    It contains no tensor or payload cache and cannot authorize skipped payload
    validation. A freshly opened store must explicitly restore before committing.
    """

    def _binding(self):
        super()._binding()
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("V8 checkpoint requires its exact authored stream schema")
        continuation = self.manifest.get("continuation", {})
        if (continuation.get("initial_memory_slots") != INITIAL_FACTS
                or continuation.get("baseline_prior_documents") != 36):
            raise ValueError("V8 continuation must retain its authenticated factual/reference base")
        documents = self.manifest.get("documents")
        if not isinstance(documents, list) or len(documents) != 24:
            raise ValueError("V8 requires exactly24 frozen observed sources")
        by_id, symbols, admitted = {}, 0, 0
        for document in documents:
            _keys(document, DOCUMENT_KEYS)
            name = document["document_id"]
            raw = _raw(document["text_hex"], SOURCE_LIMIT)
            if (not isinstance(name, str) or not name or name in by_id
                    or hashlib.sha256(raw).hexdigest() != _sha(document["sha256"])
                    or type(document["neural_tape_admission"]) is not bool
                    or document["source_kind"] not in {"factual_note", "prose", "python"}):
                raise ValueError("V8 source identity, raw bytes or admission flag changed")
            by_id[name] = document
            symbols += len(raw) + 1
            admitted += (len(raw) + 1) * document["neural_tape_admission"]
        if symbols > SOURCE_SYMBOL_LIMIT or INITIAL_FACTS + admitted > 2048:
            raise ValueError("V8 whole source observation or factual admission exceeds its bound")
        scheduled, prefix, occupied = [], [], INITIAL_FACTS
        for episode in self.manifest["episodes"]:
            ids = episode.get("append_document_ids")
            if (not isinstance(ids, list) or len(ids) > 1
                    or any(name not in by_id for name in ids)):
                raise ValueError("a V8 route boundary observes zero or one frozen source")
            prefix.append(occupied)
            scheduled.extend(ids)
            if ids and by_id[ids[0]]["neural_tape_admission"]:
                occupied += len(bytes.fromhex(by_id[ids[0]]["text_hex"])) + 1
        if len(scheduled) != 24 or set(scheduled) != set(by_id):
            raise ValueError("V8 sources must each appear once in the fixed episode schedule")
        self._v8_documents = by_id
        self._v8_fact_prefix = tuple(prefix)
        self._v8_reference_symbol_limit = INITIAL_FACTS + symbols
        self._add("checkpoint_v8_manifest_source_validations", 1)
        self._add("checkpoint_v8_manifest_source_bytes", symbols - len(documents))

    def _baseline_values(self, record):
        return _baseline_values(record, document_limit=SOURCE_LIMIT,
                                symbol_limit=self._v8_reference_symbol_limit)

    def _restore_baseline(self, record):
        return _restore_baseline(record, document_limit=SOURCE_LIMIT,
                                 symbol_limit=self._v8_reference_symbol_limit)

    def _events(self, previous, frontiers, intent):
        events = super()._events(previous, frontiers, intent)
        if "attempts" in events:
            route = ROUTES[intent["cursor"]["route_index"]]
            if route in NEURAL_ROUTES and events["attempts"][0]["prediction"] is not None:
                context = getattr(self, "_v8_commit_context", None)
                if previous is None or context is None:
                    raise ValueError("neural acknowledgement needs its owned scalar frontier")
                old = {name: (rng, slow)
                       for name, rng, slow in self._v8_controller_frontier[1]}
                current = context[2][route]
                if (current["rng_sha256"] == old[route][0]
                        or (route == "learning_disabled"
                            and current["slow_sha256"] != old[route][1])):
                    raise ValueError("actual response did not resolve RNG and disabled slow state")
                self._add("checkpoint_v8_response_resolution_validations", 1)
        if "ingests" not in events:
            return events
        if previous is None or getattr(self, "_v8_commit_context", None) is None:
            raise ValueError("V8 source validation needs its actual owned commit actors")
        models, states, proposed_hashes = self._v8_commit_context
        row = events["ingests"][0]
        route = ROUTES[intent["cursor"]["route_index"]]
        episode_index = intent["cursor"]["episode_index"]
        ids = self.manifest["episodes"][episode_index]["append_document_ids"]
        if len(ids) != 1:
            raise ValueError("V8 append intent must identify exactly one source")
        document = self._v8_documents[ids[0]]
        raw = bytes.fromhex(document["text_hex"])
        neural = route in NEURAL_ROUTES
        admitted = neural and document["neural_tape_admission"]
        if (row.get("schema") != "persistent-language-action-ingest-v8"
                or row.get("documents") != [document]
                or row.get("document_ids") != ids
                or type(row.get("raw_bytes")) is not int or row["raw_bytes"] != len(raw)
                or type(row.get("eos")) is not int or row["eos"] != 1
                or row.get("source_kind") != document["source_kind"]
                or row.get("neural_tape_admission") is not document["neural_tape_admission"]
                or row.get("fact_admitted") is not admitted
                or row.get("reference_retained") is not (not neural)
                or row.get("learning_enabled") is not (route != "learning_disabled")
                or row.get("writes_enabled") is not admitted):
            raise ValueError("source row differs from its frozen raw observation or route gate")
        if not neural:
            if (any(row.get(name) is not None for name in HASH_FIELDS)
                    or row.get("memory_used_before") is not None
                    or row.get("memory_used_after") is not None
                    or row.get("source_feedback") is not None):
                raise ValueError("simple source rows cannot claim neuronal state or publications")
            return events
        for name in HASH_FIELDS:
            _sha(row.get(name))
        old = previous["routes"][route]
        before = self._v8_fact_prefix[episode_index]
        after = before + (len(raw) + 1 if admitted else 0)
        if (row["model_before_sha256"] != old["model_sha256"]
                or row["memory_before_sha256"] != old["memory_sha256"]
                or _integer(row.get("memory_used_before")) != before
                or _integer(row.get("memory_used_after")) != after
                or states[route].used != after
                or (not admitted and row["memory_after_sha256"] != old["memory_sha256"])
                or (admitted and row["memory_after_sha256"] == old["memory_sha256"])):
            raise ValueError("source memory changed outside its exact declared factual admission")
        previous_hashes = {name: {"rng_sha256": rng, "slow_sha256": slow}
                           for name, rng, slow in self._v8_controller_frontier[1]}
        prior_controller, current_controller = previous_hashes[route], proposed_hashes[route]
        if (row["controller_rng_before_sha256"] != prior_controller["rng_sha256"]
                or row["controller_slow_before_sha256"] != prior_controller["slow_sha256"]
                or row["controller_rng_after_sha256"] != current_controller["rng_sha256"]
                or row["controller_slow_after_sha256"] != current_controller["slow_sha256"]
                or current_controller["rng_sha256"] == prior_controller["rng_sha256"]
                or (route == "learning_disabled"
                    and current_controller["slow_sha256"] != prior_controller["slow_sha256"])):
            raise ValueError("source RNG publication or disabled slow-state custody differs")
        feedback = row.get("source_feedback")
        if (not isinstance(feedback, dict) or feedback.get("document_count") != 1
                or feedback.get("raw_bytes") != len(raw)
                or feedback.get("symbol_count") != len(raw) + 1
                or feedback.get("fact_admitted") is not admitted
                or feedback.get("writes_enabled") is not admitted
                or feedback.get("learning_enabled") is not (route != "learning_disabled")
                or feedback.get("memory_before_sha256") != row["memory_before_sha256"]
                or feedback.get("memory_after_sha256") != row["memory_after_sha256"]
                or feedback.get("controller", {}).get("rng_published") is not True
                or feedback.get("controller", {}).get("learning_enabled")
                is not (route != "learning_disabled")):
            raise ValueError("source receipt does not describe the completed causal transaction")
        steps = feedback.get("source_steps")
        if (not isinstance(steps, list) or len(steps) != len(raw) + 1
                or [step.get("observed_token") for step in steps] != [*raw, 256]
                or any(step.get("category") != "scan" or step.get("prior_token") is not None
                       or step.get("retrieved_index") is not None
                       or step.get("status_offer") is not False for step in steps)):
            raise ValueError("source receipt did not consume exactly its frozen causal symbols")
        # Nested source model identities are reader-only; the parent commit binds
        # top-level full-actor after hashes to actual numerical state separately.
        from humanbrain.neuronal_splice_learner import _digest

        if feedback.get("model_after_sha256") != _digest(
                models[route].reader.state_dict(), self.work):
            raise ValueError("nested source feedback does not bind the actual reader")
        self._add("checkpoint_v8_source_event_validations", 1)
        return events

    def commit(self, models, states, baseline, host, frontiers):
        started = time.perf_counter()
        self._require_open()
        _keys(models, NEURAL_ROUTES)
        _keys(states, NEURAL_ROUTES)
        if getattr(self, "_v8_commit_context", None) is not None:
            raise ValueError("V8 checkpoint commit cannot be reentered")
        initial = self._intent is not None and self._intent[0]["kind"] == "initialize"
        old = getattr(self, "_v8_controller_frontier", None)
        if not initial and (old is None or self._intent is None
                            or old[0] != self._intent[0]["previous_sha256"]):
            raise ValueError("freshly opened V8 store must explicitly restore its actual actors")
        hashes = {} if initial else {
            name: {"rng_sha256": rng, "slow_sha256": slow} for name, rng, slow in old[1]}
        current_route = None if initial else ROUTES[self._intent[0]["cursor"]["route_index"]]
        for route in NEURAL_ROUTES:
            if initial or route == current_route:
                hashes[route] = controller_hashes(models[route].reader, self.work)
        prepared = tuple((name, hashes[name]["rng_sha256"], hashes[name]["slow_sha256"])
                         for name in NEURAL_ROUTES)
        self._v8_commit_context = models, states, hashes
        try:
            result = super().commit(models, states, baseline, host, frontiers)
        finally:
            self._v8_commit_context = None
        # A failed parent publication never advances this scalar frontier.
        self._v8_controller_frontier = result["boundary_sha256"], prepared
        result["parent_elapsed_seconds"] = result["elapsed_seconds"]
        result["elapsed_seconds"] = time.perf_counter() - started
        return result

    def restore_into(self, models):
        started = time.perf_counter()
        result = super().restore_into(models)
        hashes = {route: controller_hashes(models[route].reader, self.work)
                  for route in NEURAL_ROUTES}
        self._v8_controller_frontier = result["boundary_sha256"], tuple(
            (name, hashes[name]["rng_sha256"], hashes[name]["slow_sha256"])
            for name in NEURAL_ROUTES)
        result["parent_elapsed_seconds"] = result["elapsed_seconds"]
        result["elapsed_seconds"] = time.perf_counter() - started
        result["work"] = dict(self.work)
        return result
