"""Stable comparison source binding over existing V8 transaction custody."""

from __future__ import annotations

import hashlib

from experiments.r4_integrated.persistent_language_action_v8_checkpoint import (
    DOCUMENT_KEYS,
    SOURCE_LIMIT,
    PersistentLanguageActionV8Checkpoint,
)
from experiments.r4_integrated.persistent_session_checkpoint import (
    PersistentSessionCheckpoint,
    _keys,
    _raw,
    _sha,
)
from experiments.r4_integrated.stable_language_action_credit_workload_v1 import (
    ROSTER,
    SCHEMA,
)


class StableLanguageActionCreditCheckpoint(PersistentLanguageActionV8Checkpoint):
    """No relaxed V8 open; stable manifests have their own explicit source contract."""

    def _binding(self):
        PersistentSessionCheckpoint._binding(self)
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("stable checkpoint requires its separate comparison schema")
        continuation = self.manifest.get("continuation", {})
        if (continuation.get("initial_memory_slots") != 1982
                or continuation.get("baseline_prior_documents") != 60
                or continuation.get("baseline_prior_source_symbols") != 11498
                or continuation.get("baseline_prior_edit_demonstrations") != 49):
            raise ValueError("stable continuation changed its retained V8 factual/reference base")
        if [row["episode_id"] for row in self.manifest["episodes"]] != ROSTER:
            raise ValueError("stable checkpoint changed the fixed episode order")
        documents = self.manifest.get("documents")
        if not isinstance(documents, list) or len(documents) != 24:
            raise ValueError("stable stream needs all24 exposed sources")
        by_id, symbols = {}, 0
        for document in documents:
            _keys(document, DOCUMENT_KEYS)
            name = document["document_id"]
            raw = _raw(document["text_hex"], SOURCE_LIMIT)
            if (name != f"N{36 + len(by_id)}"
                    or hashlib.sha256(raw).hexdigest() != _sha(document["sha256"])
                    or document["neural_tape_admission"] is not False
                    or document["source_kind"] not in {"factual_note", "prose", "python"}):
                raise ValueError("stable source order, bytes or transient admission changed")
            by_id[name] = document
            symbols += len(raw) + 1
        scheduled = [name for episode in self.manifest["episodes"]
                     for name in episode["append_document_ids"]]
        if (symbols != 9884 or scheduled != list(by_id)
                or any(len(row["append_document_ids"]) > 1 for row in self.manifest["episodes"])):
            raise ValueError("stable source exposure differs from its fixed schedule")
        self._v8_documents = by_id
        self._v8_fact_prefix = (1982,) * 48
        self._v8_reference_symbol_limit = 11498 + symbols
        self._add("checkpoint_stable_manifest_source_validations", 1)
        self._add("checkpoint_stable_manifest_source_bytes", symbols - len(documents))
