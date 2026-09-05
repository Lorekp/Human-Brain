"""Authenticated after-A restoration for one fixed rehearsal follow-up.

This is an experiment-specific input/checkpoint loader, not a general persistence
API. It imports no learned state from after B, runs no encoder or predictor, and
preserves the original reader mode. A no-further-learning branch receives no
observe calls; changing its mode would change the authenticated checkpoint.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARENT_SOURCE = "dd2bd7a43f25450931d36409cd1badd1d557dd76"
PARENT_PATH = "benchmarks/real_input_qualification_v1.json"
FIXTURE_PATH = "experiments/r4_integrated/real_input_qualification_cases_v1.json"
COMPLETION_PATH = "benchmarks/real_input_qualification_v1.completion.json"
LAUNCH_PATH = "benchmarks/real_input_qualification_v1.launch.json"
PINS = {
    PARENT_PATH: "073382d23ebc66a217f81873abca7d4e93f7b7cf6b7e34c89f4b185cadd60c11",
    COMPLETION_PATH: "7837d2823f2273d949dc6fac64ebdafe254c90ca18950c032e9c4960316d32cf",
    LAUNCH_PATH: "8d54411fc99d338235e17efdbc5bacbd79afc96094d0327a516537471f5a8668",
    FIXTURE_PATH: "aa988986d8f04d7e796e485ccea33f24c96cadad09f03779de844905ee5ad41b",
}
CHECKPOINT_ARCHIVE_SHA256 = "921c1f253ade90afcc1c1c65534fcca0be0cd367dee0f334c5828dd3406ebb35"
INPUT_ARCHIVE_SHA256 = "5d60b2dad367a0a756c54fdce812b68865d51597b9b8ebfb92543e1968e12f9b"
AFTER_A = {
    "trainable": "7080a1c6a3ec9a5c45a36fc42b1c48d08c73f8dc6a310391a9f73b265dbeb383",
    "frozen_producer": "c57946b4e5173f77244d986d0e6d7c956af98114d48c637bf395b509e8559c49",
    "conventional": "9331604546f38a4cbb221ade813931a49d62178b340dc8d6600f655502c7619f",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(parent_report, arm):
    _require(arm in AFTER_A, "unsupported source arm")
    snapshot = parent_report["checkpoints"]["after_A"][arm]
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
    _require(identity == snapshot["checkpoint_sha256"] == AFTER_A[arm], "after-A identity differs")
    return snapshot


def load_state():
    """Return fixture, ID->(query,inventory,gold), report, arrays, custody.

    Returned archive arrays are immutable byte-backed copies of only three
    after-A source states. Restored readers take independent writable copies.
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
        parent["fixture"]["path"] == FIXTURE_PATH
        and parent["fixture"]["sha256"] == PINS[FIXTURE_PATH],
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
        for arm in AFTER_A:
            snapshot = _snapshot(parent, arm)
            for name, expected in snapshot["array_sha256"].items():
                key = f"after_A_{arm}_{name}"
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
        "checkpoint_archive": dict(parent["checkpoint_archive"]),
        "saved_input_archive": dict(encoded),
        "after_A": dict(AFTER_A),
        "saved_input_count": len(cache),
        "input_vector_bytes": sum(t.vectors.nbytes for t in cache.values()),
        "after_A_archive_arrays": len(archive),
        "after_A_archive_array_bytes": sum(a.nbytes for a in archive.values()),
        "parent_offers_per_source_arm": {a: _snapshot(parent, a)["offer_count"] for a in AFTER_A},
        "scope": (
            "Previously exposed TRAIN; original after-A weights and Adam state. "
            "No after-B state loaded; no encoder, predictions, gradients, updates "
            "or actions executed by loader."
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
        ConventionalAttentionReader if source_arm == "conventional" else QueryAlignedRateCellReader
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
            value = archive[f"after_A_{source_arm}_{key}"]
            _require(
                value.dtype == target[name].dtype
                and value.shape == target[name].shape
                and sha256(value.tobytes()).hexdigest() == snapshot["array_sha256"][key],
                "restoration array differs: " + key,
            )
            target[name] = value.copy()
    steps = archive[f"after_A_{source_arm}_adam_steps"]
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
