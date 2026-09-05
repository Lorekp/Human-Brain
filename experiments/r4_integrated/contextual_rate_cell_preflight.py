"""One source-bound synthetic cost preflight; never loads comparison targets."""

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

ROOT = Path(__file__).resolve().parents[2]
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
SOURCE_PATHS = (
    "src/humanbrain/rate_cell_reader.py",
    "src/humanbrain/frozen_context_encoder.py",
    "src/humanbrain/native.py",
    "experiments/r4_integrated/contextual_rate_cell_preflight.py",
    "experiments/r4_integrated/contextual_rate_cell_protocol.md",
    "experiments/r4_integrated/contextual_rate_cell_python_session.py",
    "experiments/r4_integrated/contextual_full_choice_python_session.py",
    "experiments/r4_integrated/contextual_pairwise_python_session.py",
    "experiments/r4_integrated/contextual_evidence_python_session.py",
    "experiments/r4_integrated/contextual_occurrence_motor.py",
    "experiments/r4_integrated/neutral_document_python_session.py",
    "benchmarks/native_scaling.py",
)


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(output: Path, model_dir: Path, numerical_receipt: Path, numerical_sha256: str) -> dict:
    if output.exists():
        raise FileExistsError("preflight artifacts are immutable")
    started = time.perf_counter()
    report = {"status": "failed", "started_unix": time.time(), "synthetic_only": True}
    try:
        if any(os.environ.get(name) != "1" for name in THREADS):
            raise RuntimeError("set all three numerical thread variables to 1 before Python")
        if (
            os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
            or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve()
            != ROOT / ".native-cache"
        ):
            raise RuntimeError("explicit CPU and checkout native-cache are required")
        report["environment"] = {
            name: os.environ[name]
            for name in (*THREADS, "HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        }
        report["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        report["source_sha256"] = {name: file_hash(ROOT / name) for name in SOURCE_PATHS}
        report["source_git_attestation"] = {}
        for name, expected in report["source_sha256"].items():
            committed = subprocess.check_output(
                ["git", "show", f"{report['source_revision']}:{name}"], cwd=ROOT
            )
            working = (ROOT / name).read_bytes()
            normalization = "exact" if working == committed else "working_CRLF_to_git_LF"
            if working != committed and (
                name != "experiments/r4_integrated/contextual_full_choice_python_session.py"
                or working.replace(b"\r\n", b"\n") != committed
            ):
                raise ValueError("preflight source differs from the frozen commit")
            report["source_git_attestation"][name] = {
                "raw_sha256": expected,
                "git_sha256": sha256(committed).hexdigest(),
                "normalization": normalization,
            }
        if file_hash(numerical_receipt) != numerical_sha256:
            raise ValueError("independent numerical receipt hash mismatch")
        numerical = json.loads(numerical_receipt.read_text(encoding="utf-8"))
        if (
            numerical.get("status") != "passed"
            or numerical.get("source_sha256", {}).get(SOURCE_PATHS[0])
            != report["source_sha256"][SOURCE_PATHS[0]]
        ):
            raise ValueError("independent numerical QA must pass on this exact core source")
        for name, expected in numerical.get("receipt_sha256", {}).items():
            if file_hash(ROOT / name) != expected:
                raise ValueError("referenced independent numerical receipt changed")
        report["numerical_receipt"] = {
            "path": str(numerical_receipt.resolve()),
            "sha256": numerical_sha256,
            "receipt": numerical,
        }

        import numpy as np

        from benchmarks.native_scaling import _process_memory
        from experiments.r4_integrated import contextual_evidence_python_session as shared
        from experiments.r4_integrated.contextual_pairwise_python_session import _motor
        from experiments.r4_integrated.contextual_rate_cell_python_session import retained_size
        from humanbrain.frozen_context_encoder import FrozenContextEncoder
        from humanbrain.native import loaded_build_info
        from humanbrain.rate_cell_reader import (
            RateCellReader,
            SpanInventory,
            TokenDocument,
            TokenInput,
        )

        def measured(name, operation):
            before = time.perf_counter()
            value = operation()
            seconds = time.perf_counter() - before
            report.setdefault("timings", {})[name] = seconds
            memory = _process_memory()
            report.setdefault("memory_samples", {})[name] = memory
            if time.perf_counter() - started > 900 or memory["peak_working_set_mib"] > 2048:
                raise RuntimeError("numerical phase time or parent RSS cap exceeded")
            return value, seconds

        rng = np.random.default_rng(1)

        def tokens(prefix, count):
            words = [f"{prefix}w{i}".encode() for i in range(count)]
            raw = b" ".join(words)
            ends = np.cumsum([len(word) + 1 for word in words]) - 1
            spans = tuple(
                (int(end - len(word)), int(end)) for word, end in zip(words, ends, strict=True)
            )
            vectors = rng.standard_normal((count, 384))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            return TokenInput(raw, spans, vectors, "synthetic-unit-vectors-seed-1")

        def inventory(documents, words):
            return SpanInventory(
                tuple(
                    TokenDocument(
                        f"synthetic-{documents}-{i}".encode(),
                        b"synthetic-cost-only",
                        tokens(f"d{i}", words),
                    )
                    for i in range(documents)
                )
            )

        query = tokens("query", 12)
        train, _ = measured("construct_train", lambda: inventory(16, 34))
        grown, _ = measured("construct_grown", lambda: inventory(20, 37))
        if (len(train.candidates), len(grown.candidates)) != (9520, 14060):
            raise AssertionError("synthetic inventory was reduced")
        report["inventory"] = {
            "train": 9520,
            "grown": 14060,
            "query_words": 12,
            "vector_seed": 1,
            "train_sha256": train.sha256,
            "grown_sha256": grown.sha256,
            "growth_scope": "separate larger cost envelope, not an append-only learning trial",
        }
        models = {
            mode: RateCellReader(mode="frozen" if mode == "external" else mode)
            for mode in ("trainable", "frozen_producer", "bypass", "frozen", "external")
        }
        predictions = {}
        for label, source in (("train", train), ("grown", grown)):
            predictions[label] = {}
            for mode, model in models.items():
                before = model.checkpoint_sha256
                method = model.external_only if mode == "external" else model.predict
                _, seconds = measured(
                    f"predict_{label}_{mode}",
                    lambda method=method, source=source: method(query, source),
                )
                if model.checkpoint_sha256 != before:
                    raise AssertionError("synthetic inference mutated learned state")
                predictions[label][mode] = seconds
        offers, offer_receipts = {}, []
        for mode in ("trainable", "frozen_producer", "bypass", "frozen"):
            samples = []
            for kind, gold in (("supported", (0,)), ("stop", None)):
                receipt, seconds = measured(
                    f"offer_{mode}_{kind}",
                    lambda mode=mode, gold=gold: models[mode].observe(query, train, gold),
                )
                samples.append(seconds)
                offer_receipts.append(receipt)
            offers[mode] = max(samples)
        report["synthetic_offer_receipts"] = offer_receipts
        encoder, encoder_setup = measured(
            "encoder_load", lambda: FrozenContextEncoder(model_dir, max_tokens=256, threads=1)
        )
        encoding_costs = []
        for label, count in (("query", 12), ("document", 37)):
            raw = ("ordinary " * count).strip().encode()
            encoded, seconds = measured(f"encoder_{label}", lambda raw=raw: encoder.encode(raw))
            if encoded.truncated:
                raise AssertionError("synthetic encoder timing input was truncated")
            encoding_costs.append(seconds)
        report["encoder"] = encoder.metadata
        work = Path(tempfile.mkdtemp(prefix="rate-cell-cost-", dir=ROOT / ".pytest-tmp"))
        report["synthetic_execution_workspace"] = str(work)
        motor, motor_setup = measured("motor_setup", lambda: _motor(work, report))
        case = shared.Case(
            "synthetic-roundtrip",
            "preflight",
            "one two three four five six",
            "synthetic_module",
            "stale",
        )
        before = case.files()

        def roundtrip():
            checkpoint = motor.checkpoint_sha256
            action = motor.apply(b"Configure preflight", before, case.value.encode())
            execution = shared.execute(case, action.files, before, work / "roundtrip")
            if (
                not action.edited
                or not execution["passed"]
                or motor.checkpoint_sha256 != checkpoint
            ):
                raise AssertionError("authentic synthetic motor/Python roundtrip failed")
            return execution

        report["motor_roundtrip"], execution_cost = measured("motor_and_execution", roundtrip)
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to(ROOT / ".native-cache"):
            raise RuntimeError("motor loaded a native library outside the checkout cache")
        report["native"] = {
            "source_digest": native.source_digest,
            "library_path": str(native.library_path),
            "library_sha256": file_hash(native.library_path),
        }
        encoder_bytes = sum(row["bytes"] for row in encoder.metadata["files"].values())
        payload = retained_size(models, train, grown, query, motor, report) + encoder_bytes
        report["retained_payload_bytes"] = payload
        report["payload_scope"] = (
            "synthetic objects and their arrays counted once, encoder file payload; "
            "parent RSS includes native/ONNX allocations; runtime independently "
            "counts growing lesson ledgers"
        )
        components = {
            "initialization_and_encoding": encoder_setup + 148 * max(encoding_costs) + motor_setup,
            "real_executions": 2816 * execution_cost,
            "teaching_predictions": 256 * sum(predictions["train"].values()),
            "evaluation_predictions": 128 * sum(sum(row.values()) for row in predictions.values()),
            "updater_diagnostics": 128 * sum(row["trainable"] for row in predictions.values()),
            "offers": 511
            * (
                sum(offers[mode] for mode in ("trainable", "frozen_producer", "bypass"))
                + 2 * offers["frozen"]
            ),
        }
        report["projection_seconds_by_component"] = components
        report["projected_acquisition_seconds"] = sum(components.values())
        report["projection_limitations"] = (
            "Single synthetic timing per path, max supported/STOP offer; not a runtime guarantee. "
            "No source or candidate reduction; actual run enforces wall/RSS/payload caps. "
            "Child execution wall time included, child memory not measured here. "
            "Reporting/filesystem and growing JSON ledger overhead remain outside this estimate."
        )
        if report["projected_acquisition_seconds"] > 1800 or payload > 1024**3:
            raise RuntimeError("projected acquisition or retained payload exceeds cap")
        if (
            report["source_sha256"] != {name: file_hash(ROOT / name) for name in SOURCE_PATHS}
            or file_hash(numerical_receipt) != numerical_sha256
        ):
            raise AssertionError("source or numerical attestation changed during preflight")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    report["elapsed_seconds"] = time.perf_counter() - started
    if report["elapsed_seconds"] > 900:
        report["status"] = "failed"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    parser.add_argument("--numerical-receipt", type=Path, required=True)
    parser.add_argument("--numerical-sha256", required=True)
    args = parser.parse_args()
    result = run(args.output, args.model_dir, args.numerical_receipt, args.numerical_sha256)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in ("status", "elapsed_seconds", "projected_acquisition_seconds", "error")
            },
            allow_nan=False,
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
