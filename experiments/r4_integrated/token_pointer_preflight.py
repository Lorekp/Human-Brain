"""One frozen disposable 16-question fit; it never initializes the broad run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from benchmarks.native_scaling import _process_memory
from experiments.r4_integrated.token_pointer_data import OLD, load_small_fit
from experiments.r4_integrated.token_pointer_runtime import (
    PointerPolicy,
    local_examples,
    score_counts,
    score_local,
    state_hash,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "benchmarks/token_pointer_small_fit_v1.json"
COMPLETION = ROOT / "benchmarks/token_pointer_small_fit_v1.completion.json"
ADMISSION = ROOT / "benchmarks/token_pointer_small_fit_v1.admission.json"
WORK = ROOT / ".runtime/token-pointer-small-fit-v1"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
SOURCES = (
    "src/humanbrain/token_pointer_reader.py",
    "experiments/r4_integrated/token_pointer_runtime.py",
    "experiments/r4_integrated/token_pointer_data.py",
    "experiments/r4_integrated/token_pointer_preflight.py",
    "experiments/r4_integrated/public_qa_data.py",
    "experiments/r4_integrated/real_input_qualification_cases_v1.json",
    "tests/test_token_pointer_reader.py",
    "benchmarks/native_scaling.py",
    "docs/research/token-pointer-acquisition.md",
    "pyproject.toml",
)
QA_RECEIPTS = (
    ".pytest-tmp/token-pointer-core-handoff.json",
    ".pytest-tmp/token-pointer-runtime-handoff.json",
    ".pytest-tmp/token-pointer-optional-dependency.json",
    ".pytest-tmp/token-pointer-action-qa.json",
    ".pytest-tmp/token-pointer-data-qa.json",
    ".pytest-tmp/dev/doctor-y3fqy06n/receipt.json",
    ".pytest-tmp/dev/qa-yjmt9qgd/receipt.json",
    ".pytest-tmp/token-pointer-dependency-bytes.json",
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, record):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def environment():
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            n: version(n)
            for n in ("torch", "numpy", "sympy", "networkx", "mpmath", "onnxruntime", "tokenizers")
        },
        "variables": {
            n: os.environ.get(n) for n in (*THREADS, "HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def source_identity(revision, paths):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != revision:
        raise ValueError("exact frozen Git revision required")
    hashes = {}
    for name in paths:
        raw = (ROOT / name).read_bytes()
        if raw != subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT):
            raise ValueError("working bytes differ from frozen Git source: " + name)
        hashes[name] = sha256(raw).hexdigest()
    return {"revision": actual, "files": hashes}


def prepare():
    if ADMISSION.exists() or OUTPUT.exists() or COMPLETION.exists() or WORK.exists():
        raise FileExistsError("preserve the original preflight admission/artifacts")
    fixture, inputs = load_small_fit()
    cases = fixture["cases"]
    if len(cases) != 16 or sum(r["is_impossible"] for r in cases) != 4:
        raise ValueError("original 12-supported/4-STOP roster required")
    admission = {
        "status": "prepared_without_fit",
        "fixture_sha256": file_hash(OLD),
        "ids": [r["id"] for r in cases],
        "inputs": {key: word.sha256 for key, word in sorted(inputs.items())},
        "source_files": {name: file_hash(ROOT / name) for name in SOURCES},
        "qa_receipts": {name: file_hash(ROOT / name) for name in QA_RECEIPTS},
        "branches": ["cell", "lstm"],
        "steps_per_branch": 256,
        "batch_size": 16,
        "presentations": 8192,
        "readonly_probes": 64,
        "seconds": 900,
        "peak_parent_mib": 4096,
        "criterion": {"each_branch_final_supported": 12, "each_branch_final_STOP": 4},
        "exposure": "Original exposed supervised TRAIN sixteen; O11 role-label caveat unchanged",
        "state_use": "Disposable; never seed broad comparison",
        "environment": environment(),
        "disk": {
            "free_bytes": shutil.disk_usage(ROOT).free,
            "artifact_allowance_bytes": 64 * 2**20,
            "reserve_bytes": 2**30,
        },
    }
    if admission["disk"]["free_bytes"] - 64 * 2**20 < 2**30:
        raise OSError("small-fit artifact allowance would breach disk reserve")
    write_json(ADMISSION, admission)
    print(json.dumps({"admission": str(ADMISSION), "sha256": file_hash(ADMISSION)}), flush=True)


def run(revision, admission_sha256):
    if OUTPUT.exists() or COMPLETION.exists() or WORK.exists():
        raise FileExistsError("one fit only; preserve previous outputs")
    if file_hash(ADMISSION) != admission_sha256:
        raise ValueError("exact frozen admission required")
    admission = json.loads(ADMISSION.read_bytes())
    source = source_identity(revision, (*SOURCES, ADMISSION.relative_to(ROOT).as_posix()))
    if any(source["files"][name] != value for name, value in admission["source_files"].items()):
        raise ValueError("source differs from numerical preflight admission")
    if any(file_hash(ROOT / name) != value for name, value in admission["qa_receipts"].items()):
        raise ValueError("preflight QA or dependency receipt differs")
    env = environment()
    if (
        env != admission["environment"]
        or env["torch_threads"] != 1
        or any(os.environ.get(n) != "1" for n in THREADS)
        or torch.__version__ != "2.14.0+cpu"
    ):
        raise ValueError("single-thread qualified CPU environment differs")
    WORK.mkdir()
    began = time.perf_counter()
    result = {
        "status": "failed",
        "started_unix": time.time(),
        "source": source,
        "admission_sha256": admission_sha256,
        "environment": env,
        "branches": {},
    }
    checkpoints = {}

    def guard(label):
        memory = _process_memory()
        result["last_memory"] = {"label": label, **memory}
        if time.perf_counter() - began > 900:
            raise TimeoutError("fixed fifteen-minute small-fit budget exhausted")
        if memory["peak_working_set_mib"] > 4096:
            raise MemoryError("fixed 4 GiB parent peak limit exceeded")
        if shutil.disk_usage(ROOT).free < 2**30:
            raise OSError("disk reserve breached")

    try:
        fixture, inputs = load_small_fit()
        if {key: word.sha256 for key, word in sorted(inputs.items())} != admission["inputs"]:
            raise ValueError("qualified original word inputs differ")
        cases, documents = fixture["cases"], fixture["documents"]
        queries, contexts, gold = local_examples(cases, documents, inputs.__getitem__)
        for kind in ("cell", "lstm"):
            guard(kind + ":initial")
            policy = PointerPolicy(kind)
            branch = result["branches"][kind] = {
                "initial_state_sha256": policy.checkpoint_sha256,
                "parameter_count": sum(p.numel() for p in policy.reader.parameters()),
                "initial": score_local(policy, cases, documents, inputs.__getitem__),
            }
            checkpoints[kind + "_initial"] = policy.checkpoint()
            durations = []
            with (WORK / (kind + "-updates.jsonl")).open(
                "x", encoding="utf-8", newline="\n"
            ) as log:
                for step in range(256):
                    guard(kind + ":" + str(step))
                    start = time.perf_counter()
                    update = policy.update_batch(queries, contexts, gold)
                    branch["steps"], branch["presentations"] = policy.steps, policy.presentations
                    elapsed = time.perf_counter() - start
                    durations.append(elapsed)
                    log.write(
                        json.dumps(
                            {
                                "step": step + 1,
                                "presentations": policy.presentations,
                                "seconds": elapsed,
                                **update,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    log.flush()
                    if (step + 1) % 64 == 0:
                        print(
                            json.dumps(
                                {
                                    "branch": kind,
                                    "steps": step + 1,
                                    "elapsed_seconds": time.perf_counter() - began,
                                }
                            ),
                            flush=True,
                        )
            guard(kind + ":final_probe")
            branch["final"] = score_local(policy, cases, documents, inputs.__getitem__)
            branch["initial_counts"] = score_counts(branch["initial"])
            branch["final_counts"] = score_counts(branch["final"])
            branch["steps"], branch["presentations"] = policy.steps, policy.presentations
            branch["update_seconds"] = {
                "sum": sum(durations),
                "median": float(np.median(durations)),
                "p95": float(np.quantile(durations, 0.95)),
                "max": max(durations),
            }
            branch["final_state_sha256"] = policy.checkpoint_sha256
            checkpoints[kind + "_final"] = policy.checkpoint()
            print(json.dumps({"branch": kind, "final": branch["final_counts"]}), flush=True)
        guard("complete_fit")
        result["qualified"] = all(
            row["final_counts"]["supported"] == 12
            and row["final_counts"]["STOP"] == 4
            and row["steps"] == 256
            and row["presentations"] == 4096
            for row in result["branches"].values()
        )
        result["status"] = "completed"
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        archive = WORK / "checkpoints.pt"
        try:
            if result["status"] != "completed" and "policy" in locals():
                checkpoints[kind + "_partial"] = policy.checkpoint()
            torch.save(checkpoints, archive)
            restored = torch.load(archive, map_location="cpu", weights_only=True)
            if state_hash(restored) != state_hash(checkpoints):
                raise ValueError("saved checkpoint payload differs")
            result["checkpoint_archive"] = {
                "path": archive.relative_to(ROOT).as_posix(),
                "bytes": archive.stat().st_size,
                "sha256": file_hash(archive),
            }
            result["journal_files"] = {
                p.relative_to(ROOT).as_posix(): file_hash(p) for p in WORK.glob("*-updates.jsonl")
            }
            guard("archive_written_and_restored")
        except BaseException:
            result["status"] = "failed"
            result["finalization_error"] = traceback.format_exc()
        result["elapsed_seconds"] = time.perf_counter() - began
        result["final_memory"] = _process_memory()
        result["finalization"] = (
            "Requires matching passed completion receipt; scores alone insufficient"
        )
        write_json(OUTPUT, result)
        try:
            guard("report_written")
        except BaseException:
            result["status"] = "failed"
            result["report_finalization_error"] = traceback.format_exc()
            write_json(OUTPUT, result)
        completion = {
            "status": "passed" if result["status"] == "completed" else "failed",
            "qualified": result["status"] == "completed" and result.get("qualified", False),
            "elapsed_seconds": time.perf_counter() - began,
            "final_memory": _process_memory(),
            "free_disk_bytes": shutil.disk_usage(ROOT).free,
            "report_sha256": file_hash(OUTPUT),
        }
        write_json(COMPLETION, completion)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "qualified": completion["qualified"],
                    "elapsed_seconds": completion["elapsed_seconds"],
                    "report_sha256": file_hash(OUTPUT),
                }
            ),
            flush=True,
        )
    return 0 if result["status"] == "completed" and result["qualified"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--revision")
    parser.add_argument("--admission-sha256")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.revision or not args.admission_sha256:
            parser.error("run requires --revision and --admission-sha256")
        raise SystemExit(run(args.revision, args.admission_sha256))
