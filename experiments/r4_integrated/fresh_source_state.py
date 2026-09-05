"""Authenticated final-rehearsal restoration and cumulative source custody.

This is an experiment-specific input/checkpoint loader, not a general persistence
API. It loads only the three declared final rehearsal parents, runs no predictor, and
preserves the original reader mode. A no-further-learning branch receives no
observe calls; changing its mode would change the authenticated checkpoint.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.r4_integrated.rehearsal_state import _canonical, _file_hash, _require

ROOT = Path(__file__).resolve().parents[2]
PARENT_PUBLICATION = "57bd0924dd8dc38f61ad97e58fc1d0f99b15e49e"
PARENT_SOURCE = "2eb0fb19bacbf1e6b08b265840bba2b9e70f7d32"
PARENT_PATH = "benchmarks/fixed_rehearsal_v1.json"
FIXTURE_PATH = "experiments/r4_integrated/real_input_qualification_cases_v1.json"
COMPLETION_PATH = "benchmarks/fixed_rehearsal_v1.completion.json"
LAUNCH_PATH = "benchmarks/fixed_rehearsal_v1.launch.json"
PINS = {
    PARENT_PATH: "a219ece11778ae7b9ded22e1103fb33556baf50f8aebbf66e4deef4c3312779d",
    COMPLETION_PATH: "f6ec3c827009b7a1c33b99323bfd8abed1d4ec170e299ec2b04d01ecc4b135c3",
    LAUNCH_PATH: "6d2216fa6390d947268740b02ad23ad1bce161b8b18d2c297f2b3afb766fb711",
    FIXTURE_PATH: "aa988986d8f04d7e796e485ccea33f24c96cadad09f03779de844905ee5ad41b",
}
CHECKPOINT_ARCHIVE_SHA256 = "404edb1d9866331cfbdac747fd979626b262f59350895d09584b32302fa3b458"
INPUT_ARCHIVE_SHA256 = "5d60b2dad367a0a756c54fdce812b68865d51597b9b8ebfb92543e1968e12f9b"
FINAL_PARENTS = {
    "cell_rehearsal": "919bdf095c36b49aa7d77ae8d1066cb03e64bd65c8fc67a988042e04d7c92f72",
    "frozen_producer_rehearsal": "00dafc3b81181ba89a551b5ecc7e0f2a906cb8985e14b3b6c47801ffccb83ce8",
    "conventional_rehearsal": "f8e1fea767669e18ffdcc0c641a6ed057e6a049fc7124f5456d1787546785998",
}


def _snapshot(parent_report, arm):
    _require(arm in FINAL_PARENTS, "unsupported source arm")
    snapshot = parent_report["checkpoints"]["final"][arm]
    identity = sha256(
        _canonical(
            {
                "parameters": snapshot["parameters"],
                "arrays": snapshot["array_sha256"],
                "offers": snapshot["offer_count"],
                "updates": snapshot["update_count"],
            }
        ).encode()
    ).hexdigest()
    _require(
        identity == snapshot["checkpoint_sha256"] == FINAL_PARENTS[arm],
        "final-rehearsal identity differs",
    )
    return snapshot


def load_state():
    """Return fixture, ID->(query,inventory,gold), report, arrays, custody.

    Returned archive arrays are immutable byte-backed copies of only three
    final-rehearsal source states. Restored readers take independent writable copies.
    Labels in the input triples are explicit supervised feedback, not predictor
    arguments. Retained paragraphs and word vectors are unchanged external inputs.
    """
    from humanbrain.rate_cell_reader import SpanInventory, TokenDocument, TokenInput

    records = {}
    for name, expected in PINS.items():
        raw = (ROOT / name).read_bytes()
        _require(sha256(raw).hexdigest() == expected, "published artifact differs: " + name)
        records[name] = json.loads(raw)
    parent = records[PARENT_PATH]
    completion = records[COMPLETION_PATH]
    launch = records[LAUNCH_PATH]
    fixture = records[FIXTURE_PATH]
    _require(
        parent["status"] == "completed"
        and not parent.get("error")
        and not parent["finalization_errors"],
        "parent run incomplete",
    )
    _require(
        completion["status"] == "passed"
        and not completion["finalization_errors"]
        and completion["report_sha256"] == PINS[PARENT_PATH],
        "parent completion differs",
    )
    _require(
        launch["status"] == "passed"
        and launch["exit_code"] == 0
        and launch["child_completion"] == completion,
        "parent launch failed or differs",
    )
    _require(parent["source"]["revision"] == PARENT_SOURCE, "parent source revision differs")
    for name, expected in parent["source"]["files"].items():
        _require(_file_hash(ROOT / name) == expected, "parent source changed: " + name)
    _require(
        parent["custody"]["published_sha256"][FIXTURE_PATH] == PINS[FIXTURE_PATH]
        and parent["custody"]["saved_input_archive"] == fixture["encoded_inputs"],
        "parent fixture differs",
    )

    checkpoint_path = Path(parent["checkpoint_archive"]["path"])
    _require(
        parent["checkpoint_archive"]["sha256"] == CHECKPOINT_ARCHIVE_SHA256
        and _file_hash(checkpoint_path) == CHECKPOINT_ARCHIVE_SHA256,
        "parent checkpoint archive differs",
    )
    archive = {}
    with np.load(checkpoint_path, allow_pickle=False) as saved:
        for arm in FINAL_PARENTS:
            snapshot = _snapshot(parent, arm)
            for name, expected in snapshot["array_sha256"].items():
                key = f"final_{arm}_{name}"
                value = saved[key]
                _require(
                    sha256(value.tobytes()).hexdigest() == expected
                    and bool(np.isfinite(value).all()),
                    "saved checkpoint array differs: " + key,
                )
                archive[key] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(
                    value.shape
                )

    encoded = fixture["encoded_inputs"]
    input_path = ROOT / encoded["path"]
    _require(
        encoded["sha256"] == INPUT_ARCHIVE_SHA256
        and _file_hash(input_path) == INPUT_ARCHIVE_SHA256,
        "saved word vectors differ",
    )
    cache = {}
    with np.load(input_path, allow_pickle=False) as saved:
        _require(
            set(saved.files) == {r["array"] for r in fixture["saved_inputs"]},
            "saved input roster differs",
        )
        for row in fixture["saved_inputs"]:
            token = TokenInput(
                bytes.fromhex(row["raw_hex"]),
                tuple(map(tuple, row["byte_spans"])),
                saved[row["array"]],
                row["encoder_sha256"],
            )
            _require(
                token.sha256 == row["sha256"]
                and sha256(token.vectors.tobytes()).hexdigest() == row["vectors_sha256"],
                "original TokenInput differs",
            )
            cache[row["array"]] = token
    ids = fixture["blocks"]["A"] + fixture["blocks"]["B"]
    _require(
        [r["id"] for r in fixture["cases"]] == ids and len(set(ids)) == 16,
        "original lesson roster differs",
    )
    inputs = {}
    for lesson in fixture["cases"]:
        link = fixture["input_links"][lesson["id"]]
        document = fixture["documents"][lesson["paragraph_id"]]
        query, tokens = cache[link["query_array"]], cache[link["document_array"]]
        _require(
            query.raw == lesson["question"].encode() and tokens.raw == document["context"].encode(),
            "original raw input differs",
        )
        inventory = SpanInventory(
            (
                TokenDocument(
                    lesson["paragraph_id"].encode(), document["article_sha256"].encode(), tokens
                ),
            )
        )
        _require(
            inventory.sha256 == link["original_paragraph_inventory_sha256"],
            "original candidate competition differs",
        )
        gold = None
        if not lesson["is_impossible"]:
            wanted = {(a["byte_start"], a["byte_end"]) for a in lesson["answers"]}
            for answer in lesson["answers"]:
                _require(
                    tokens.raw[answer["byte_start"] : answer["byte_end"]]
                    == answer["text"].encode(),
                    "original annotation byte slice differs",
                )
            gold = tuple(
                i
                for i, c in enumerate(inventory.candidates)
                if (c.byte_start, c.byte_end) in wanted
            )
            _require(
                bool(gold)
                and {
                    (inventory.candidates[i].byte_start, inventory.candidates[i].byte_end)
                    for i in gold
                }
                == wanted,
                "every original annotation must remain reachable",
            )
        inputs[lesson["id"]] = (query, inventory, gold)
    custody = {
        "published_sha256": dict(PINS),
        "parent_source": parent["source"],
        "parent_publication": PARENT_PUBLICATION,
        "checkpoint_archive": dict(parent["checkpoint_archive"]),
        "saved_input_archive": dict(encoded),
        "final": dict(FINAL_PARENTS),
        "saved_input_count": len(cache),
        "input_vector_bytes": sum(t.vectors.nbytes for t in cache.values()),
        "final_archive_arrays": len(archive),
        "final_archive_array_bytes": sum(a.nbytes for a in archive.values()),
        "parent_offers_per_source_arm": {
            a: _snapshot(parent, a)["offer_count"] for a in FINAL_PARENTS
        },
        "scope": (
            "Previously exposed TRAIN; original final-rehearsal weights and Adam state. "
            "Only the three final rehearsal states loaded; no encoder, predictions, "
            "gradients, updates or actions executed by loader."
        ),
    }
    return fixture, inputs, parent, archive, custody


def restore_core(parent_report, archive, source_arm):
    """Restore one independent original-mode core and demand its exact snapshot."""
    from humanbrain.conventional_attention_reader import ConventionalAttentionReader
    from humanbrain.query_aligned_rate_cell_reader import QueryAlignedRateCellReader

    snapshot = _snapshot(parent_report, source_arm)
    parameters = snapshot["parameters"]
    constructor = (
        ConventionalAttentionReader
        if source_arm == "conventional_rehearsal"
        else QueryAlignedRateCellReader
    )
    core = constructor(
        **{k: parameters[k] for k in ("mode", "seed", "learning_rate", "gradient_clip_norm")}
    )
    _require(
        _canonical(core.parameters) == _canonical(parameters),
        "current reader configuration differs",
    )
    names = tuple(parameters["parameter_names"])
    for prefix, target in (
        ("weight_", core._weights),
        ("adam_first_", core._first),
        ("adam_second_", core._second),
    ):
        for name in names:
            key = prefix + name
            value = archive[f"final_{source_arm}_{key}"]
            _require(
                value.dtype == target[name].dtype
                and value.shape == target[name].shape
                and sha256(value.tobytes()).hexdigest() == snapshot["array_sha256"][key],
                "restoration array differs: " + key,
            )
            target[name] = value.copy()
    steps = archive[f"final_{source_arm}_adam_steps"]
    _require(
        steps.dtype == np.dtype("int64")
        and steps.shape == (len(names),)
        and sha256(steps.tobytes()).hexdigest() == snapshot["array_sha256"]["adam_steps"],
        "restoration counters differ",
    )
    core._steps = {name: int(step) for name, step in zip(names, steps, strict=True)}
    core.offer_count, core.update_count = snapshot["offer_count"], snapshot["update_count"]
    _require(
        _canonical(core.snapshot()) == _canonical(snapshot), "restored complete snapshot differs"
    )
    return core


def source_gold(lesson, inventory):
    """Evaluator-only annotation mapping; equal bytes elsewhere are not gold."""
    document_id = lesson["paragraph_id"].encode()
    matches = [d for d in inventory.documents if d.document_id == document_id]
    _require(len(matches) == 1, "original source must occur exactly once")
    if lesson["is_impossible"]:
        _require(not lesson["answers"], "STOP lesson has supported annotations")
        return None
    raw = matches[0].tokens.raw
    wanted = set()
    for answer in lesson["answers"]:
        left, right = answer["byte_start"], answer["byte_end"]
        _require(
            0 <= left < right <= len(raw) and raw[left:right] == answer["text"].encode(),
            "official annotation does not match original UTF-8 source",
        )
        wanted.add((left, right))
    gold = tuple(
        i
        for i, c in enumerate(inventory.candidates)
        if inventory.documents[c.document_index].document_id == document_id
        and (c.byte_start, c.byte_end) in wanted
    )
    _require(
        bool(gold)
        and {(inventory.candidates[i].byte_start, inventory.candidates[i].byte_end) for i in gold}
        == wanted,
        "every official source occurrence must remain reachable",
    )
    return gold


class SourcePool:
    """Cumulative raw inputs with one current pool and single-source oracle caches.

    No labels enter TokenInput, TokenDocument, or SpanInventory. Gold is produced
    only by the explicit evaluator-facing inputs/oracle calls. The caller adds
    freshly encoded inputs during the scored session, and owns introduction order.
    Older pooled inventories are not retained here. The optional full document
    order authenticates every append, including the initial sixteen sources.
    """

    def __init__(self, old_fixture, old_inputs, *, expected_order=None):
        from humanbrain.rate_cell_reader import SpanInventory

        self._documents = {}
        self._queries = {}
        self._oracles = {}
        self._inventory = None
        self._expected_order = tuple(expected_order) if expected_order is not None else None
        self.serial = 0
        self.encoder_sha256 = None
        for lesson in old_fixture["cases"]:
            query, original, gold = old_inputs[lesson["id"]]
            _require(len(original.documents) == 1, "old oracle is not one original paragraph")
            document = original.documents[0]
            self.add_document(
                lesson["paragraph_id"],
                old_fixture["documents"][lesson["paragraph_id"]],
                document.tokens,
            )
            self.add_query(lesson, query)
            _require(source_gold(lesson, original) == gold, "old annotation custody differs")
            _require(
                original.sha256 == self._oracles[lesson["paragraph_id"]].sha256,
                "old original-source inventory changed",
            )
            self._oracles[lesson["paragraph_id"]] = original
        self.serial = 0
        self._inventory = SpanInventory(tuple(self._documents.values()))

    def _check_encoder(self, tokens):
        if self.encoder_sha256 is None:
            self.encoder_sha256 = tokens.encoder_sha256
        _require(tokens.encoder_sha256 == self.encoder_sha256, "input encoder identity differs")

    def add_document(self, paragraph_id, record, tokens):
        from humanbrain.rate_cell_reader import SpanInventory, TokenDocument

        _require(paragraph_id not in self._documents, "source already introduced")
        _require(tokens.raw == record["context"].encode(), "document raw bytes differ")
        _require(
            tokens.byte_spans == tuple(map(tuple, record["word_byte_spans"])),
            "document word custody differs",
        )
        if self._expected_order is not None:
            index = len(self._documents)
            _require(
                index < len(self._expected_order) and paragraph_id == self._expected_order[index],
                "cumulative document order differs",
            )
        self._check_encoder(tokens)
        document = TokenDocument(paragraph_id.encode(), record["article_sha256"].encode(), tokens)
        oracle = SpanInventory((document,))
        self._documents[paragraph_id] = document
        self._oracles[paragraph_id] = oracle
        self._inventory = None
        self.serial += 1

    def add_query(self, lesson, tokens):
        _require(lesson["id"] not in self._queries, "question already introduced")
        _require(tokens.raw == lesson["question"].encode(), "original question bytes differ")
        self._check_encoder(tokens)
        self._queries[lesson["id"]] = tokens

    @property
    def document_ids(self):
        return tuple(self._documents)

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def inventory(self):
        from humanbrain.rate_cell_reader import SpanInventory

        if self._inventory is None:
            self._inventory = SpanInventory(tuple(self._documents.values()))
        return self._inventory

    def inputs(self, lesson):
        query = self._queries[lesson["id"]]
        _require(query.raw == lesson["question"].encode(), "question identity differs")
        inventory = self.inventory
        return query, inventory, source_gold(lesson, inventory)

    def oracle(self, lesson):
        query = self._queries[lesson["id"]]
        _require(query.raw == lesson["question"].encode(), "question identity differs")
        inventory = self._oracles[lesson["paragraph_id"]]
        return query, inventory, source_gold(lesson, inventory)

    def manifest(self):
        """Current inference inventory metadata, without task IDs or annotations."""
        inventory = self.inventory
        canonical = _canonical(inventory.candidate_manifest()).encode("utf-8")
        return {
            "serial": self.serial,
            "inventory_sha256": inventory.sha256,
            "document_ids": list(self.document_ids),
            "document_count": self.document_count,
            "window_count": len(inventory.windows),
            "candidate_count": len(inventory.candidates),
            "candidate_manifest_sha256": sha256(canonical).hexdigest(),
            "candidate_manifest_canonical_bytes": len(canonical),
        }

    def storage_summary(self):
        """Payload lower bound; recursive caller accounting includes Python overhead."""
        tokens = [d.tokens for d in self._documents.values()] + list(self._queries.values())
        unique_tokens = {id(t): t for t in tokens}
        inventories = {id(self.inventory): self.inventory}
        inventories.update({id(v): v for v in self._oracles.values()})
        return {
            "scope": "retained raw/vector/span payload; excludes Python object overhead",
            "token_inputs": len(unique_tokens),
            "raw_bytes": sum(len(t.raw) for t in unique_tokens.values()),
            "vector_bytes": sum(t.vectors.nbytes for t in unique_tokens.values()),
            "local_span_array_bytes": sum(
                a.nbytes for v in inventories.values() for a in v.local_spans
            ),
            "candidate_byte_payload": sum(
                len(c.emitted) for v in inventories.values() for c in v.candidates
            ),
            "retained_inventory_count": len(inventories),
        }
