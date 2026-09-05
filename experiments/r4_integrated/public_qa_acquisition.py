"""One public-QA acquisition comparison and updates-disabled Python return."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.r4_integrated.contextual_rate_cell_python_session import (
    ExecutedReader as AuthorityReader,
)
from experiments.r4_integrated.contextual_rate_cell_python_session import (
    digest,
)
from experiments.r4_integrated.public_qa_data import file_hash, load_selection
from experiments.r4_integrated.rate_cell_session_support import retained_size
from humanbrain.conventional_attention_reader import ConventionalAttentionReader
from humanbrain.query_aligned_rate_cell_reader import QueryAlignedRateCellReader
from humanbrain.rate_cell_reader import SpanInventory, TokenDocument, TokenInput, _word_spans

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "experiments/r4_integrated/public_qa_selection_v1.json"
SELECTION_SHA256 = "72fa7ac873643be36701c486a66c24ae67631fcd95c6fe18c2021539dd3cfe76"
FIXTURE = ROOT / "experiments/r4_integrated/contextual_full_choice_train.json"
FIXTURE_SHA256 = "9c1055c8c1fbfc3f5e92d997c5e335913619962c6aa47f0bdf9b984a005f4059"
ARMS = ("trainable", "frozen_producer", "frozen", "conventional", "external")
UPDATING = ("trainable", "frozen_producer", "conventional")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
VECTOR_CAP = 768 * 1024**2
SOURCE_PATHS = (
    "experiments/r4_integrated/public_qa_acquisition.py",
    "experiments/r4_integrated/public_qa_data.py",
    "experiments/r4_integrated/public_qa_selection_v1.json",
    "docs/research/public-qa-acquisition-plan.md",
    "src/humanbrain/conventional_attention_reader.py",
    "src/humanbrain/query_aligned_rate_cell_reader.py",
    "src/humanbrain/rate_cell_reader.py",
    "src/humanbrain/frozen_context_encoder.py",
    "experiments/r4_integrated/rate_cell_session_support.py",
    "experiments/r4_integrated/contextual_rate_cell_python_session.py",
    "experiments/r4_integrated/contextual_pairwise_python_session.py",
    "experiments/r4_integrated/contextual_full_choice_python_session.py",
    "experiments/r4_integrated/contextual_full_choice_train.json",
    "experiments/r4_integrated/contextual_evidence_python_session.py",
    "experiments/r4_integrated/contextual_occurrence_motor.py",
    "experiments/r4_integrated/neutral_document_python_session.py",
    "src/humanbrain/opaque_document_event_tape.py",
    "src/humanbrain/multiscale_opaque_event_tape.py",
    "src/humanbrain/unified_opaque_observation_tape.py",
    "src/humanbrain/knowledge_conditioned_occurrence_substitution.py",
    "src/humanbrain/variability_aware_segment_binder.py",
    "src/humanbrain/cells.py",
    "src/humanbrain/resting_glif_race.py",
    "src/humanbrain/native.py",
    "benchmarks/native_scaling.py",
)


class MotorReturnReader(AuthorityReader):
    """Bind a trained core to the existing action path, with no feedback authority."""

    def __init__(self, core, mode, inventory, queries):
        if mode not in ARMS or any(raw != tokens.raw for raw, tokens in queries.items()):
            raise ValueError("invalid motor-return mode or query cache")
        self.core, self.mode, self.inventory = core, mode, inventory
        self.queries = MappingProxyType(dict(queries))
        self.feedback_root = digest({"scope": "public_qa_motor_return_inference_only"})
        self.training_pool = None


def new_cores():
    cores = {name: QueryAlignedRateCellReader(mode=name) for name in ARMS[:3]}
    cores["conventional"] = ConventionalAttentionReader()
    cores["external"] = cores["frozen"]
    initial = cores["trainable"].checkpoint_arrays()
    for name in ("frozen_producer", "frozen"):
        current = cores[name].checkpoint_arrays()
        if current.keys() != initial.keys() or any(
            not np.array_equal(initial[k], current[k]) for k in initial
        ):
            raise ValueError("cell control initialization differs")
    return cores


def source_identity(revision):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != revision:
        raise ValueError("exact frozen source revision required")
    files = {}
    for name in SOURCE_PATHS:
        raw = (ROOT / name).read_bytes()
        if raw != subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT):
            raise ValueError(f"working source differs from frozen Git bytes: {name}")
        files[name] = sha256(raw).hexdigest()
    return {"revision": actual, "files": files}


def gold_indices(lesson, inventory):
    if lesson["is_impossible"]:
        return None
    spans = {(a["byte_start"], a["byte_end"]) for a in lesson["answers"]}
    indices = tuple(
        i for i, c in enumerate(inventory.candidates) if (c.byte_start, c.byte_end) in spans
    )
    if (
        not indices
        or {(inventory.candidates[i].byte_start, inventory.candidates[i].byte_end) for i in indices}
        != spans
    ):
        raise ValueError("every original accepted annotation must remain reachable")
    return indices


def qa_prediction(core, arm, query, inventory, lesson):
    """Targets score the returned decision; they never enter predictor arguments."""
    prediction = (core.external_only if arm == "external" else core.predict)(query, inventory)
    candidate = (
        None
        if prediction.candidate_index is None
        else inventory.candidates[prediction.candidate_index]
    )
    emitted = None if candidate is None else candidate.emitted.hex()
    supported = not lesson["is_impossible"]
    indices = gold_indices(lesson, inventory)
    scores = prediction.span_scores
    best_gold = None if indices is None else float(scores[list(indices)].max())
    return {
        "id": lesson["id"],
        "arm": arm,
        "supported": supported,
        "correct": emitted in lesson["accepted_text_hex"] if supported else prediction.stopped,
        "emitted_hex": emitted,
        "stopped": prediction.stopped,
        "candidate_index": prediction.candidate_index,
        "byte_span": None if candidate is None else [candidate.byte_start, candidate.byte_end],
        "query_sha256": query.sha256,
        "inventory_sha256": inventory.sha256,
        "checkpoint_sha256": prediction.checkpoint_sha256,
        "p_stop": prediction.p_stop,
        "stop_logit": prediction.stop_logit,
        "stop_reason": prediction.stop_reason,
        "candidate_count": len(scores),
        "span_scores_sha256": sha256(scores.tobytes()).hexdigest(),
        "best_gold_rank": None
        if best_gold is None
        else int(1 + np.count_nonzero(scores > best_gold)),
        "conditional_best_is_gold": None
        if indices is None
        else bool(best_gold == scores.max() and np.count_nonzero(scores == best_gold) == 1),
    }


def qa_summary(rows, supported, unsupported):
    return {
        "completed": len(rows),
        "supported": supported,
        "unsupported": unsupported,
        "exact_supported": sum(bool(r["supported"] and r["correct"]) for r in rows),
        "correct_stop": sum(bool(not r["supported"] and r["correct"]) for r in rows),
    }


def decision(phases, motor, terminal, data, fixture):
    result = {"complete": False, "acquisition_passed": False, "motor_ready": False}
    if terminal != "completed" or set(phases) != {"early_train", "final_train", "public_probe"}:
        return result
    by_id = {r["id"]: r for r in data["train"]}
    expected = {
        "early_train": [by_id[i] for i in data["early_train_ids"]],
        "final_train": [by_id[i] for i in (*data["early_train_ids"], *data["late_train_ids"])],
        "public_probe": data["probe"],
    }
    for phase, lessons in expected.items():
        roster = [(r["id"], not r["is_impossible"]) for r in lessons]
        if set(phases[phase]) != set(ARMS) or any(
            [(r.get("id"), r.get("supported")) for r in phases[phase][arm]] != roster
            or any(r.get("arm") != arm for r in phases[phase][arm])
            for arm in ARMS
        ):
            return result
    counts = {"early_train": (24, 8), "final_train": (48, 16), "public_probe": (192, 64)}
    summaries = {
        phase: {arm: qa_summary(rows[arm], *counts[phase]) for arm in ARMS}
        for phase, rows in phases.items()
    }
    if any(
        s["completed"] != sum(counts[p]) for p, arms in summaries.items() for s in arms.values()
    ):
        return result
    motor_roster = [(t["id"], t["answer"]) for t in fixture["tasks"]]
    if set(motor) != {"actual", "alternate"} or any(
        set(motor[p]) != set(ARMS)
        or any(
            [(r.get("case"), r.get("expected_value")) for r in motor[p][a].get("rows", [])]
            != motor_roster
            for a in ARMS
        )
        for p in motor
    ):
        return result
    public = summaries["public_probe"]
    trained, conventional = public["trainable"], public["conventional"]
    fit = summaries["final_train"]["trainable"]
    early = summaries["early_train"]["trainable"]
    final_rows = phases["final_train"]["trainable"]
    first, last = qa_summary(final_rows[:32], 24, 8), qa_summary(final_rows[32:], 24, 8)
    checks = {
        "public_floor": trained["exact_supported"] >= 96 and trained["correct_stop"] >= 52,
        "learned_advantage": all(
            trained["exact_supported"] >= public[a]["exact_supported"] + 20
            for a in ("frozen", "frozen_producer")
        ),
        "conventional": trained["exact_supported"] >= conventional["exact_supported"]
        and trained["correct_stop"] >= conventional["correct_stop"] - 3,
        "fit_floor": fit["exact_supported"] >= 36 and fit["correct_stop"] >= 13,
        "fit_strata": all(
            s["exact_supported"] >= 18 and s["correct_stop"] >= 6 for s in (first, last)
        ),
        "stability": first["exact_supported"] >= early["exact_supported"] - 2
        and first["correct_stop"] >= early["correct_stop"] - 1,
    }
    motor_ready = all(
        all(r["execution"]["decoys_unchanged"] for r in motor[p]["trainable"]["rows"])
        and sum(
            r["expected_value"] is not None
            and r["execution"]["passed"]
            and r["execution"]["decoys_unchanged"]
            for r in motor[p]["trainable"]["rows"]
        )
        >= 44
        and sum(
            r["expected_value"] is None
            and r["correct_no_action"]
            and r["execution"]["decoys_unchanged"]
            for r in motor[p]["trainable"]["rows"]
        )
        >= 15
        for p in motor
    )
    return {
        "complete": True,
        "acquisition_passed": all(checks.values()),
        "motor_ready": motor_ready,
        "checks": checks,
        "qa": summaries,
        "scope": "bounded supervised acquisition/motor readiness, not the general integrated gate",
    }


def preflight_cost(data, fixture, encoder, cores, guard):
    """Price fixed maximum shapes with synthetic vectors/text, without target inference."""
    rng = np.random.default_rng(0)

    def synthetic(count):
        raw = " ".join(f"word{i}" for i in range(count)).encode()
        vectors = rng.standard_normal((count, 384))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return TokenInput(raw, _word_spans(raw), vectors, "synthetic-cost-only")

    qwords = max(r["question_word_count"] for split in ("train", "probe") for r in data[split])
    dwords = max(d["word_count"] for d in data["documents"].values())
    query = synthetic(qwords)
    source = TokenDocument(b"synthetic", b"cost", synthetic(dwords))
    started = time.perf_counter()
    inventory = SpanInventory((source,))
    build_seconds = time.perf_counter() - started
    motor_query_words = max(
        len(_word_spans(q.encode()))
        for q in (*(t["query"] for t in fixture["tasks"]), *fixture["alternate_queries"].values())
    )
    motor_query = synthetic(motor_query_words)
    motor_inventory = SpanInventory(
        tuple(
            TokenDocument(
                f"synthetic-{i}".encode(), b"cost", synthetic(len(_word_spans(d["text"].encode())))
            )
            for i, d in enumerate(fixture["documents"])
        )
    )
    timings = {}
    for arm, core in cores.items():
        predict = core.external_only if arm == "external" else core.predict
        started = time.perf_counter()
        predict(query, inventory)
        public_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predict(motor_query, motor_inventory)
        motor_seconds = time.perf_counter() - started
        values = {"public_predict_seconds": public_seconds, "motor_predict_seconds": motor_seconds}
        if arm in UPDATING:
            for name, gold in (("supported", (0,)), ("unsupported", None)):
                started = time.perf_counter()
                core.observe(query, inventory, gold)
                values[f"{name}_observe_seconds"] = time.perf_counter() - started
        timings[arm] = values
        guard(f"synthetic:{arm}")
    started = time.perf_counter()
    encoded = encoder.encode(b"the " * 254)
    if encoded.token_count != 256 or encoded.truncated:
        raise ValueError("neutral encoding cost must exercise the full 256-token limit")
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    subprocess.run(
        [os.sys.executable, "-c", "print('synthetic execution cost')"],
        check=True,
        capture_output=True,
        timeout=20,
    )
    child_seconds = time.perf_counter() - started
    raw_inputs = {d["context"].encode() for d in data["documents"].values()}
    raw_inputs.update(r["question"].encode() for s in ("train", "probe") for r in data[s])
    raw_inputs.update(d["text"].encode() for d in fixture["documents"])
    raw_inputs.update(t["query"].encode() for t in fixture["tasks"])
    raw_inputs.update(q.encode() for q in fixture["alternate_queries"].values())
    components = {
        "encoding": len(raw_inputs) * encode_seconds,
        "inventory_builds": (1024 + 32 + 64 + 256) * build_seconds,
        "qa_predictions": (1024 + 32 + 64 + 256)
        * sum(v["public_predict_seconds"] for v in timings.values()),
        "offers": 2047
        * sum(
            max(timings[a]["supported_observe_seconds"], timings[a]["unsupported_observe_seconds"])
            for a in UPDATING
        ),
        "motor_predictions": 128 * sum(v["motor_predict_seconds"] for v in timings.values()),
        "child_executions": 642 * child_seconds,
        "motor_and_custody_reserve": 120.0,
        "artifact_write_reserve": 120.0,
    }
    return {
        "synthetic_only": True,
        "query_words": qwords,
        "document_words": dwords,
        "unique_encoding_inputs": len(raw_inputs),
        "timings": timings,
        "encode_256_tokens_seconds": encode_seconds,
        "python_child_seconds": child_seconds,
        "components_seconds": components,
        "projected_acquisition_seconds": sum(components.values()),
        "projected_token_vector_bytes": sum(len(_word_spans(raw)) * 384 * 8 for raw in raw_inputs),
        "scope": (
            "maximum public shapes, original motor shapes, one sample each; "
            "conservative estimate, not measured full run"
        ),
    }


def run(mode, output, revision, receipt_path, receipt_sha256):
    completion_path = output.with_suffix(".completion.json")
    if any(p.exists() for p in (output, output.with_suffix(".npz"), completion_path)):
        raise FileExistsError("prior artifacts cannot be overwritten")
    if (
        any(os.environ.get(t) != "1" for t in THREADS)
        or os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
    ):
        raise RuntimeError("set CPU and all three numerical thread variables before Python")
    if Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve() != ROOT / ".native-cache":
        raise RuntimeError("set the checkout native cache before Python")
    started, started_unix = time.perf_counter(), time.time()
    identity = source_identity(revision)
    if file_hash(receipt_path) != receipt_sha256:
        raise ValueError("admission receipt hash differs")
    admission = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mode == "preflight":
        if (
            admission.get("status") != "passed"
            or admission.get("source_files") != identity["files"]
        ):
            raise ValueError("source-bound preparation QA is required")
        combined_start = started_unix
    else:
        if (
            admission.get("status") != "passed"
            or admission.get("mode") != "preflight"
            or admission["source"] != identity
        ):
            raise ValueError("passing same-source preflight is required")
        completion = json.loads(
            receipt_path.with_suffix(".completion.json").read_text(encoding="utf-8")
        )
        if (
            completion.get("status") != "passed"
            or completion.get("report_sha256") != receipt_sha256
        ):
            raise ValueError("preflight artifact finalization must pass and match its report")
        combined_start = admission["started_unix"]
    from benchmarks.native_scaling import _process_memory
    from experiments.r4_integrated import contextual_evidence_python_session as shared
    from experiments.r4_integrated.contextual_full_choice_python_session import _item, _summary
    from experiments.r4_integrated.contextual_pairwise_python_session import _motor
    from humanbrain.frozen_context_encoder import FrozenContextEncoder
    from humanbrain.native import loaded_build_info

    (ROOT / ".runtime").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="public-qa-", dir=ROOT / ".runtime"))
    report = {
        "mode": mode,
        "status": "failed",
        "terminal": "failed",
        "source": identity,
        "started_unix": started_unix,
        "combined_started_unix": combined_start,
        "admission": {"path": str(receipt_path), "sha256": receipt_sha256, "receipt": admission},
        "workspace_root": str(work),
        "training": [],
        "phases": {},
        "motor_return": {},
        "input_records": [],
        "events": [],
        "limits": {
            "combined_seconds": 7200,
            "preflight_seconds": 900,
            "peak_parent_mib": 2048,
            "token_vector_bytes": VECTOR_CAP,
        },
        "environment": {
            k: os.environ.get(k) for k in (*THREADS, "HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "python": {"version": os.sys.version, "executable": os.sys.executable},
        "scope": (
            "Supervised public TRAIN QA; paragraph-visible probes; exposed Python "
            "transfer of training; no official/protected evaluation"
        ),
        "completion_receipt": str(completion_path),
        "finalization": "Report status is provisional; the completion receipt is authoritative.",
    }
    arrays, cache, cores = {}, {}, {}
    encoder, error, vector_bytes = None, None, 0
    data = fixture = None

    def guard(label):
        memory = _process_memory()
        report["last_memory"] = {"label": label, **memory}
        if memory["peak_working_set_mib"] > 2048:
            raise MemoryError("peak parent working set exceeds 2 GiB")
        if vector_bytes > VECTOR_CAP:
            raise MemoryError("retained token vector cache exceeds 768 MiB")
        if time.time() - combined_start > 7200 or (
            mode == "preflight" and time.perf_counter() - started > 900
        ):
            raise TimeoutError("fixed public-QA time budget exhausted")

    def event(name, **values):
        value = {"event": name, "elapsed_seconds": time.perf_counter() - started, **values}
        report["events"].append(value)
        with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value), flush=True)

    def checkpoint(name):
        report.setdefault("checkpoints", {})[name] = {a: c.snapshot() for a, c in cores.items()}
        for arm in ARMS[:4]:
            arrays.update(
                {f"{name}_{arm}_{k}": v for k, v in cores[arm].checkpoint_arrays().items()}
            )

    def encode(raw):
        nonlocal vector_bytes
        if raw not in cache:
            guard("encoding")
            encoded = encoder.encode(raw)
            tokens = TokenInput.from_encoding(raw, encoded)
            name = f"input_{len(cache):04d}"
            cache[raw], arrays[name] = tokens, tokens.vectors
            vector_bytes += tokens.vectors.nbytes
            report["input_records"].append(
                {"array": name, **tokens.record(), "encoder_token_count": encoded.token_count}
            )
            guard("encoded")
        return cache[raw]

    try:
        guard("before_load")
        data = load_selection(SELECTION, SELECTION_SHA256)
        if file_hash(FIXTURE) != FIXTURE_SHA256:
            raise ValueError("original motor fixture differs")
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        report["selection"] = {
            "path": str(SELECTION),
            "sha256": SELECTION_SHA256,
            "summary": data["summary"],
            "cache_estimates": data["cache_estimates"],
        }
        report["motor_fixture_sha256"] = FIXTURE_SHA256
        began = time.perf_counter()
        encoder = FrozenContextEncoder(ROOT / ".models/all-MiniLM-L6-v2", max_tokens=256, threads=1)
        report["encoder_load_seconds"], report["encoder"] = (
            time.perf_counter() - began,
            encoder.metadata,
        )
        report["encoder_sha256"] = encoder.sha256
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to(ROOT / ".native-cache"):
            raise RuntimeError("native motor library is outside the checkout cache")
        report["native_motor"] = {
            "source_digest": native.source_digest,
            "library": str(native.library_path),
            "sha256": file_hash(native.library_path),
        }
        cores = new_cores()
        guard("after_load")
        if mode == "preflight":
            report["cost"] = preflight_cost(data, fixture, encoder, cores, guard)
            if report["cost"]["projected_token_vector_bytes"] > VECTOR_CAP:
                raise MemoryError("projected complete token-vector cache exceeds 768 MiB")
            projected = (
                report["cost"]["projected_acquisition_seconds"] + report["encoder_load_seconds"]
            )
            report["projected_acquisition_seconds"] = projected
            if time.time() - combined_start + projected > 7200:
                raise TimeoutError(
                    "projected full acquisition exceeds the two-hour combined budget"
                )
        else:
            if (
                encoder.sha256 != admission["encoder_sha256"]
                or report["native_motor"] != admission["native_motor"]
            ):
                raise ValueError("encoder or native motor identity differs from preflight")
            began = time.perf_counter()
            for d in data["documents"].values():
                encode(d["context"].encode())
            for split in ("train", "probe"):
                for lesson in data[split]:
                    encode(lesson["question"].encode())
            report["public_encoding_seconds"] = time.perf_counter() - began
            event("public_inputs_encoded", inputs=len(cache), vector_bytes=vector_bytes)
            checkpoint("initial")

            def inputs(lesson):
                doc = data["documents"][lesson["paragraph_id"]]
                query = cache[lesson["question"].encode()]
                source = TokenDocument(
                    lesson["paragraph_id"].encode(),
                    doc["article_sha256"].encode(),
                    cache[doc["context"].encode()],
                )
                inventory = SpanInventory((source,))
                return query, inventory, gold_indices(lesson, inventory)

            def probe(name, lessons):
                before = {a: c.checkpoint_sha256 for a, c in cores.items()}
                rows = report["phases"][name] = {a: [] for a in ARMS}
                for lesson in lessons:
                    query, inventory, _ = inputs(lesson)
                    report["active_request"] = {"phase": name, "id": lesson["id"]}
                    for arm, core in cores.items():
                        rows[arm].append(qa_prediction(core, arm, query, inventory, lesson))
                        guard(f"{name}:{arm}")
                if before != {a: c.checkpoint_sha256 for a, c in cores.items()}:
                    raise RuntimeError("read-only QA probe changed a core")
                report.pop("active_request", None)
                event(
                    name,
                    completed=len(lessons),
                    scores={
                        a: qa_summary(
                            v,
                            sum(not r["is_impossible"] for r in lessons),
                            sum(r["is_impossible"] for r in lessons),
                        )
                        for a, v in rows.items()
                    },
                )

            by_id = {r["id"]: r for r in data["train"]}
            previous = None
            for step, lesson in enumerate(data["train"], 1):
                query, inventory, gold = inputs(lesson)
                row = {"encounter": step, "id": lesson["id"], "first": {}, "offers": []}
                report["training"].append(row)
                report["active_request"] = {
                    "phase": "acquisition",
                    "encounter": step,
                    "id": lesson["id"],
                }
                for arm, core in cores.items():
                    row["first"][arm] = qa_prediction(core, arm, query, inventory, lesson)
                    guard(f"first:{arm}")
                for arm in UPDATING:
                    for kind, item in (
                        ("current", (lesson["id"], query, inventory, gold)),
                        ("previous", previous),
                    ):
                        if item is None:
                            continue
                        item_id, q, inv, target = item
                        row["offers"].append(
                            {
                                "arm": arm,
                                "kind": kind,
                                "id": item_id,
                                "receipt": cores[arm].observe(q, inv, target),
                            }
                        )
                        guard(f"offer:{arm}")
                previous = (lesson["id"], query, inventory, gold)
                report.pop("active_request", None)
                if step % 128 == 0:
                    event(
                        "acquisition",
                        encounters=step,
                        offers={a: cores[a].offer_count for a in UPDATING},
                    )
                if step == 256:
                    checkpoint("after_256")
                    probe("early_train", [by_id[i] for i in data["early_train_ids"]])
            if (
                any(cores[a].offer_count != 2047 for a in UPDATING)
                or cores["frozen"].offer_count != 0
            ):
                raise RuntimeError("fixed training offer count differs")
            previous = None
            checkpoint("trained")
            probe(
                "final_train",
                [by_id[i] for i in (*data["early_train_ids"], *data["late_train_ids"])],
            )
            probe("public_probe", data["probe"])
            documents = tuple(
                TokenDocument(
                    d["id"].encode(), b"rate-cell-public-train-v1", encode(d["text"].encode())
                )
                for d in fixture["documents"]
            )
            motor_queries = {
                q.encode(): encode(q.encode())
                for q in (
                    *(t["query"] for t in fixture["tasks"]),
                    *fixture["alternate_queries"].values(),
                )
            }
            inventory = SpanInventory(documents)
            if len(inventory.candidates) != 8162:
                raise ValueError("original full-source motor inventory differs")
            report["motor_inventory"] = {
                "sha256": inventory.sha256,
                "candidates": inventory.candidate_manifest(),
            }
            motor = _motor(work, report)
            guard("motor_demonstrations")
            adapters = {
                a: MotorReturnReader(c, a, inventory, motor_queries) for a, c in cores.items()
            }
            before = {a: c.checkpoint_sha256 for a, c in cores.items()}
            for wording in ("actual", "alternate"):
                phase = report["motor_return"][wording] = {}
                for arm, adapter in adapters.items():
                    rows = []
                    phase[arm] = {"rows": rows}
                    for task in fixture["tasks"]:
                        query = (
                            task["query"]
                            if wording == "actual"
                            else fixture["alternate_queries"][task["id"]]
                        )
                        report["active_request"] = {
                            "phase": "motor",
                            "wording": wording,
                            "arm": arm,
                            "id": task["id"],
                        }
                        row, _ = shared.attempt(
                            adapter,
                            "respond",
                            motor,
                            _item(task, query),
                            work / wording / arm / task["id"],
                        )
                        row["source_storage"] = (
                            "retained raw documents and normalized word vectors; "
                            "span candidates reference original byte intervals"
                        )
                        rows.append(row)
                        guard(f"motor:{arm}")
                    phase[arm]["summary"] = _summary(rows, fixture["tasks"])
                    event("motor_return", wording=wording, arm=arm, summary=phase[arm]["summary"])
            if before != {a: c.checkpoint_sha256 for a, c in cores.items()}:
                raise RuntimeError("motor return changed a trained core")
            report.pop("active_request", None)
            report["retained_payload"] = {
                "python_numpy_core_cache_motor_bytes": retained_size(
                    cores, cache, motor, inventory
                ),
                "raw_public_source_and_label_bytes": retained_size(data),
                "token_vector_bytes": vector_bytes,
            }
        report["source_after"] = source_identity(revision)
        for name, value in encoder.metadata["files"].items():
            if file_hash(ROOT / ".models/all-MiniLM-L6-v2" / name) != value["sha256"]:
                raise ValueError("encoder file changed during the run")
        if file_hash(native.library_path) != report["native_motor"]["sha256"]:
            raise ValueError("native motor file changed during the run")
        guard("authenticated")
        report["terminal"] = "completed"
        report["status"] = "passed" if mode == "preflight" else "completed"
    except Exception as exc:
        error = exc
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:

        def finalization_error(stage, exc):
            nonlocal error
            detail = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            report.setdefault("finalization_errors", []).append(detail)
            if error is None:
                error = exc
                report["error"] = detail
            report.update(status="failed", terminal="failed")

        try:
            if cores and mode != "preflight":
                checkpoint("terminal")
            output.parent.mkdir(parents=True, exist_ok=True)
            if arrays:
                target = output.with_suffix(".npz")
                with target.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                report["arrays"] = {
                    "path": str(target),
                    "sha256": file_hash(target),
                    "bytes": target.stat().st_size,
                }
            guard("after_checkpoint_write")
        except Exception as exc:
            finalization_error("checkpoint_export", exc)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["elapsed_scope"] = (
            "through checkpoint output; completion receipt measures main JSON serialization, "
            "write and hashing too, excluding only its own small write and Python bootstrap"
        )
        report["token_vector_bytes"] = vector_bytes
        report_hash = None
        try:
            if mode != "preflight":
                report["decision"] = decision(
                    report["phases"], report["motor_return"], report["terminal"], data, fixture
                )
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    report, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                stream.write("\n")
            report_hash = file_hash(output)
            guard("after_report_write")
        except Exception as exc:
            finalization_error("report_export", exc)
        completion = {
            "status": report["status"],
            "terminal": report["terminal"],
            "report_sha256": report_hash,
            "elapsed_seconds": time.perf_counter() - started,
            "combined_elapsed_seconds": time.time() - combined_start,
            "last_memory": report.get("last_memory"),
            "error": report.get("error"),
            "finalization_errors": report.get("finalization_errors", []),
        }
        try:
            with completion_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(completion, stream, allow_nan=False)
                stream.write("\n")
        except Exception as exc:
            finalization_error("completion_receipt", exc)
            print(
                json.dumps({"completion_receipt_failed": str(exc), "error": report["error"]}),
                flush=True,
            )
        report["elapsed_seconds"] = completion["elapsed_seconds"]
    if error is not None:
        raise error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "acquisition"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    result = run(args.mode, args.output, args.expected_revision, args.receipt, args.receipt_sha256)
    print(
        json.dumps(
            {
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "decision": result.get("decision"),
                "projected_acquisition_seconds": result.get("projected_acquisition_seconds"),
            }
        )
    )


if __name__ == "__main__":
    main()
