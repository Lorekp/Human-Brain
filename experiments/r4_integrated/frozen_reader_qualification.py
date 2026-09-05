"""One fixed external-reader ceiling check; no learning, actions or protected data."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict, is_dataclass
from hashlib import file_digest
from pathlib import Path

from humanbrain.frozen_extractive_reader import FrozenExtractiveReader
from humanbrain.opaque_document_event_tape import OpaqueDocument

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "experiments/r4_integrated/contextual_full_choice_train.json"
FIXTURE_SHA256 = "9c1055c8c1fbfc3f5e92d997c5e335913619962c6aa47f0bdf9b984a005f4059"
MODEL_DIR = ROOT / ".models/roberta-base-squad2-optimum-db07770"
SOURCE_PATHS = (
    "src/humanbrain/frozen_extractive_reader.py",
    "src/humanbrain/opaque_document_event_tape.py",
    "experiments/r4_integrated/frozen_reader_qualification.py",
    "experiments/r4_integrated/frozen_reader_qualification_protocol.md",
)
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
GROUPS = (
    "known_source:actual",
    "known_source:alternate",
    "full_corpus:actual",
    "full_corpus:alternate",
)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def process_memory() -> dict:
    """Same Windows counters as native_scaling, without importing native models."""
    if os.name != "nt":
        raise RuntimeError("this qualification's peak-memory contract is Windows-specific")
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.argtypes, current.restype = [], wintypes.HANDLE
    measure = ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    measure.restype = wintypes.BOOL
    if not measure(current(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return {
        "working_set_mib": counters.WorkingSetSize / 1024**2,
        "peak_working_set_mib": counters.PeakWorkingSetSize / 1024**2,
        "private_memory_mib": counters.PrivateUsage / 1024**2,
    }


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def source_identity(expected_revision: str) -> dict:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != expected_revision:
        raise ValueError("qualification requires its exact frozen source revision")
    hashes = {}
    for name in SOURCE_PATHS:
        raw = (ROOT / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        if raw != committed:
            raise ValueError(f"working source differs from frozen Git bytes: {name}")
        hashes[name] = file_hash(ROOT / name)
    return {"revision": revision, "files": hashes}


def summarize(rows: list[dict], terminal: str) -> dict:
    groups = {}
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {
            "completed": len(selected),
            "supported": 48,
            "unsupported": 16,
            "exact_supported": sum(row["supported"] and row["correct"] for row in selected),
            "correct_stop": sum(not row["supported"] and row["correct"] for row in selected),
        }
    complete = (
        terminal == "completed"
        and len(rows) == 256
        and all(group["completed"] == 64 for group in groups.values())
    )
    qualifies = complete and all(
        groups[f"known_source:{wording}"]["exact_supported"] >= 44
        and groups[f"known_source:{wording}"]["correct_stop"] >= 15
        for wording in ("actual", "alternate")
    )
    return {"groups": groups, "complete": complete, "reader_qualifies": qualifies}


def run(
    mode: str, output: Path, expected_revision: str, receipt_path: Path, receipt_sha256: str
) -> dict:
    if output.exists():
        raise FileExistsError("qualification records cannot be overwritten")
    identity = source_identity(expected_revision)
    if file_hash(receipt_path) != receipt_sha256:
        raise ValueError("admission receipt hash differs")
    admission = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mode == "qualification":
        if admission.get("status") != "passed" or admission.get("mode") != "preflight":
            raise ValueError("a passing fixed-reader preflight is required")
        if admission["source"] != identity:
            raise ValueError("preflight source identity differs")
    elif admission.get("status") != "passed" or admission.get("source_files") != identity["files"]:
        raise ValueError("source-bound synthetic QA receipt is required")
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise RuntimeError("set the three numerical thread variables to 1 before Python")
    if os.environ.get("HUMANBRAIN_DEVICE") != "cpu":
        raise RuntimeError("fixed reader qualification requires CPU")

    began = time.perf_counter()
    started_unix = time.time()
    combined_start = admission["started_unix"] if mode == "qualification" else started_unix
    limit = 1800 if mode == "qualification" else 900
    report = {
        "mode": mode,
        "status": "failed",
        "terminal": "failed",
        "source": identity,
        "started_unix": started_unix,
        "combined_started_unix": combined_start,
        "admission": {"path": str(receipt_path), "sha256": receipt_sha256, "receipt": admission},
        "scope": (
            "Exposed TRAIN external-reader qualification only; no learning, Python actions "
            "or official/protected evaluation."
        ),
        "byte_encoding": (
            "bytes in nested dataclasses serialize as hexadecimal; original text is not normalized"
        ),
        "rows": [],
        "memory_observations": [],
        "environment": {name: os.environ.get(name) for name in (*THREADS, "HUMANBRAIN_DEVICE")},
        "limits": {"seconds": limit, "combined_seconds": 2700, "peak_parent_mib": 2048},
    }
    error = None
    reader = None

    def guard(label):
        memory = process_memory()
        report["memory_observations"].append({"label": label, **memory})
        if memory["peak_working_set_mib"] > 2048:
            raise MemoryError("parent peak working set exceeds 2 GiB")
        if time.perf_counter() - began > limit or time.time() - combined_start > 2700:
            raise TimeoutError("fixed qualification time budget exhausted")

    try:
        guard("before_load")
        reader = FrozenExtractiveReader(MODEL_DIR)
        report["reader"] = reader.metadata
        report["reader_sha256"] = reader.sha256
        report["load_seconds"] = reader.load_seconds
        guard("after_load")
        if mode == "preflight":
            # All text here is neutral synthetic load, never comparison targets.
            documents = tuple(
                OpaqueDocument(
                    f"synthetic-{i}".encode(),
                    f"synthetic-source-{i}".encode(),
                    (
                        f"Synthetic document {i}. "
                        + "This paragraph describes a neutral runtime sample. " * 80
                    ).encode(),
                )
                for i in range(16)
            )
            query = b"What does this synthetic paragraph describe?"
            samples = []
            for name, offered in (("single_source", documents[:1]), ("full_corpus", documents)):
                reading = reader.read(query, offered)
                report.setdefault("synthetic_readings", []).append(
                    {"condition": name, "reading": reading}
                )
                if (
                    reading.window_count <= len(offered)
                    or reading.paired_tokens < len(offered) * 386
                ):
                    raise ValueError("synthetic cost check must cover full and overflow windows")
                samples.append(reading.elapsed_seconds / reading.window_count)
                guard(name)
            report["projected_qualification_seconds"] = reader.load_seconds + max(samples) * 2176
            report["projection_scope"] = (
                "2176 original query/document pairs priced at neutral overlapping-window average; "
                "includes decoder but omits final audit JSON write"
            )
            if report["projected_qualification_seconds"] > 1800:
                raise TimeoutError("projected qualification exceeds 30 minutes")
        else:
            if file_hash(FIXTURE) != FIXTURE_SHA256:
                raise ValueError("fixed qualification fixture changed")
            fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
            documents = tuple(
                OpaqueDocument(doc["id"].encode(), doc["id"].encode(), doc["text"].encode())
                for doc in fixture["documents"]
            )
            by_id = {doc.document_id.decode(): doc for doc in documents}
            if len(documents) != 16 or len(fixture["tasks"]) != 64:
                raise ValueError("fixed qualification roster shape differs")
            report["fixture_sha256"] = FIXTURE_SHA256
            report["documents"] = [asdict(doc) for doc in documents]
            report["retained_raw_source_bytes"] = sum(len(doc.content) for doc in documents)
            if reader.sha256 != admission["reader_sha256"]:
                raise ValueError("reader identity differs from passing preflight")
            for group in GROUPS:
                access, wording = group.split(":")
                for task in fixture["tasks"]:
                    query = (
                        task["query"]
                        if wording == "actual"
                        else fixture["alternate_queries"][task["id"]]
                    )
                    offered = (
                        (by_id[task["expected_source_id"]],)
                        if access == "known_source"
                        else documents
                    )
                    report["active_request"] = {"group": group, "case": task["id"], "query": query}
                    reading = reader.read(query.encode(), offered)
                    expected = None if task["answer"] is None else task["answer"].encode()
                    correct = reading.stopped if expected is None else reading.emitted == expected
                    report["rows"].append(
                        {
                            "group": group,
                            "case": task["id"],
                            "query": query,
                            "supported": expected is not None,
                            "expected_hex": None if expected is None else expected.hex(),
                            "offered_source_ids": [doc.document_id.decode() for doc in offered],
                            "correct": bool(correct),
                            "reading": reading,
                        }
                    )
                    guard(f"{group}:{task['id']}")
                    report.pop("active_request")
                partial = summarize(report["rows"], "running")["groups"][group]
                print(json.dumps({"group": group, **partial}), flush=True)
        report["source_after"] = source_identity(expected_revision)
        for name, value in reader.metadata["files"].items():
            if file_hash(MODEL_DIR / name) != value["sha256"]:
                raise ValueError(f"model file changed during qualification: {name}")
        guard("after_authentication")
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
        report["elapsed_seconds"] = time.perf_counter() - began
        report["elapsed_scope"] = (
            "through source/model reauthentication; final JSON serialization/write excluded"
        )
        try:
            report["final_memory"] = process_memory()
        except Exception as memory_error:
            report["final_memory_error"] = str(memory_error)
        if mode == "qualification":
            report["qualification"] = summarize(report["rows"], report["terminal"])
            report["processed_windows"] = sum(row["reading"].window_count for row in report["rows"])
            report["processed_paired_tokens"] = sum(
                row["reading"].paired_tokens for row in report["rows"]
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    report,
                    stream,
                    default=json_value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
        except Exception as write_error:
            if error is not None:
                raise error from write_error
            raise
    if error is not None:
        raise error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "qualification"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    report = run(args.mode, args.output, args.expected_revision, args.receipt, args.receipt_sha256)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "qualification": report.get("qualification"),
                "projected_qualification_seconds": report.get("projected_qualification_seconds"),
            }
        )
    )


if __name__ == "__main__":
    main()
