"""One bounded host-only byte-event preservation and event-count preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from humanbrain.byte_events import ByteEventSegmenter  # noqa: E402

CONFIG = {
    "schema": "local-byte-event-preflight-v1",
    "seconds": 60,
    "parent_mib": 2048,
    "cap_bytes": 16,
    "word_bytes": "ASCII A-Z a-z 0-9 underscore",
    "whitespace_hex": "090a0b0c0d20",
    "other_bytes": "singleton",
    "schedule_lessons": 1024,
    "documents_per_lesson": 2,
    "schedule_chunkings": ["single", "bytes"],
    "fixture_all_splits_max_bytes": 64,
    "conceptual_eos_per_invocation": 1,
    "actual_neural_steps": 0,
    "aggregate_seconds_including_audit": 60,
}
PRIOR = ".pytest-tmp/local-memory-multiscale-v1-run/"
INPUTS = {
    PRIOR + "offline_teacher_roster.json": (
        "009b42e22976733bba207e46337ec23d5f6de5ab15ab061bc6fd08539e44a2b1"
    ),
    PRIOR + "completion.json": "0fe498198b397405f42824c0eacbdeae52a51a88378c1e6124b271f30733130b",
    ".pytest-tmp/local-memory-multiscale-v1-run-independent-audit.json": (
        "a9a1ebfec69f909124f8583834469739fdf13b959999ddac6b03aa0dea87dc48"
    ),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def duration_clock():
    clock = time.get_clock_info("perf_counter")
    return {
        "name": "perf_counter",
        "implementation": clock.implementation,
        "resolution_seconds": clock.resolution,
        "monotonic": clock.monotonic,
        "adjustable": clock.adjustable,
    }


def sources():
    paths = [
        Path(__file__),
        Path(__file__).with_suffix(".md"),
        ROOT / "src/humanbrain/byte_events.py",
        ROOT / "src/humanbrain/__init__.py",
        ROOT / "experiments/r4_integrated/__init__.py",
        ROOT / "docs/research/local-token-event-interface.md",
        ROOT / "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1",
    ]
    initializer = ROOT / "experiments/__init__.py"
    if initializer.exists():
        paths.append(initializer)
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def fixtures():
    items = [("empty", b""), ("all_bytes", bytes(range(256)))]
    items.extend((f"byte_{value:02x}", bytes([value])) for value in range(256))
    for length in (15, 16, 17, 31, 32, 33):
        items.extend(
            ((f"word_{length}", b"a" * length), (f"space_{length}", (b" \t" * length)[:length]))
        )
    items.extend(
        (
            ("ascii_mixed", b" A0_z\t\r\n.,!?\x00\x7f"),
            ("unicode_bytes", "e\u0301 \u00e9 \u4e2d\U0001f642".encode("utf-8")),
            ("invalid_utf8", b"\xff\xc0\xaf\xed\xa0\x80\xf0\x28\x8c\xbc"),
            ("mixed_word", b"aZ_09B7___8"),
        )
    )
    return [{"id": name, "kind": "fixture", "input_hex": raw.hex()} for name, raw in items]


def training_inputs():
    for name, expected in INPUTS.items():
        assert file_hash(ROOT / name) == expected
    complete = json.loads((ROOT / (PRIOR + "completion.json")).read_bytes())
    audit = json.loads(
        (ROOT / ".pytest-tmp/local-memory-multiscale-v1-run-independent-audit.json").read_bytes()
    )
    assert complete["status"] == audit["status"] == "passed"
    assert complete["source_revision"] == "a1ab5ee0d82c5cc047392ec8e6b22be3a4386438"
    assert (
        complete["artifacts"]["offline_teacher_roster.json"]
        == INPUTS[PRIOR + "offline_teacher_roster.json"]
    )
    assert INPUTS[PRIOR + "completion.json"] in audit["inputs"].values()
    # Intentionally access only schedule document/query strings, never labels or checks.
    groups = json.loads((ROOT / (PRIOR + "offline_teacher_roster.json")).read_bytes())["schedule"]
    assert len(groups) == 256 and all(len(group) == 4 for group in groups)
    rows = []
    for index, lesson in enumerate(lesson for group in groups for lesson in group):
        documents = lesson["document_hex"]
        assert len(documents) == 2
        for document_index, raw in enumerate(documents):
            rows.append(
                {
                    "id": f"train-{index}-doc-{document_index}",
                    "kind": "document",
                    "schedule_index": index,
                    "document_index": document_index,
                    "input_hex": raw,
                }
            )
        rows.append(
            {
                "id": f"train-{index}-query",
                "kind": "query",
                "schedule_index": index,
                "input_hex": lesson["query_hex"],
            }
        )
    assert len(rows) == 3072
    return rows


def invocation(raw, chunks, name):
    segmenter = ByteEventSegmenter()
    events = []
    position = 0
    for length in chunks:
        events.extend(segmenter.feed(raw[position : position + length]))
        position += length
    events.extend(segmenter.finish())
    assert position == segmenter.bytes_seen == len(raw)
    assert segmenter.buffered_bytes == 0 and segmenter.max_buffered_bytes <= 16
    rows = [{"start": e.start, "end": e.end, "payload_hex": e.payload.hex()} for e in events]
    cursor = 0
    for event in events:
        assert event.start == cursor and event.end == cursor + len(event.payload)
        assert 1 <= len(event.payload) <= 16 and raw[event.start : event.end] == event.payload
        cursor = event.end
    assert cursor == len(raw) and b"".join(event.payload for event in events) == raw
    return rows, {
        "name": name,
        "chunk_lengths": chunks,
        "event_digest": canonical(rows),
        "raw_bytes": len(raw),
        "raw_events": len(rows),
        "feed_calls": len(chunks),
        "finish_calls": 1,
        "max_buffered_bytes": segmenter.max_buffered_bytes,
    }


def qualify_input(row):
    raw = bytes.fromhex(row["input_hex"])
    chunkings = [("single", [len(raw)]), ("bytes", [1] * len(raw) if raw else [0])]
    if row["kind"] == "fixture" and len(raw) <= CONFIG["fixture_all_splits_max_bytes"]:
        chunkings.extend((f"split-{i}", [i, len(raw) - i]) for i in range(len(raw) + 1))
    reference, variants = None, []
    for name, chunks in chunkings:
        events, receipt = invocation(raw, chunks, name)
        if reference is None:
            reference = events
        else:
            assert reference == events, (row["id"], name)
        variants.append(receipt)
    return {**row, "events": reference, "variants": variants}


def count_invocations(variants):
    count = len(variants)
    raw = sum(v["raw_bytes"] for v in variants)
    events = sum(v["raw_events"] for v in variants)
    return {
        "segmentation_invocations": count,
        "raw_bytes_offered": raw,
        "raw_events_emitted": events,
        "feed_calls": sum(v["feed_calls"] for v in variants),
        "finish_calls": sum(v["finish_calls"] for v in variants),
        "conceptual_end_events": count,
        "conceptual_byte_baseline_steps": raw + count,
        "conceptual_controller_events": events + count,
        "max_buffered_bytes": max((v["max_buffered_bytes"] for v in variants), default=0),
    }


def run(args):
    assert args.output.is_absolute() and args.admission.is_absolute()
    assert not args.output.exists()
    assert file_hash(args.admission) == args.admission_sha
    admission = json.loads(args.admission.read_bytes())
    manifest = sources()
    assert admission["status"] == "preprocessing_qa"
    assert admission["source_revision"] == args.revision and len(args.revision) == 40
    assert admission["source_sha256"] == manifest and admission["policy"] == CONFIG
    assert admission["prior_inputs"] == INPUTS
    assert admission["aggregate_seconds_including_audit"] == 60
    assert file_hash(ROOT / admission["audit_path"]) == admission["audit_sha256"]
    assert all(math.isfinite(t) for t in (args.launch_started_unix, args.deadline_unix))
    assert 0 < args.deadline_unix - args.launch_started_unix <= 60.001

    def guard():
        assert args.launch_started_unix <= time.time() <= args.deadline_unix
        assert time.perf_counter() - START <= 60

    guard()
    args.output.mkdir()
    report = {
        "status": "failed",
        "config": CONFIG,
        "source_revision": args.revision,
        "source_sha256": manifest,
        "prior_inputs": INPUTS,
        "admission_sha256": args.admission_sha,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "launch_started_unix": args.launch_started_unix,
        "deadline_unix": args.deadline_unix,
        "duration_clock": duration_clock(),
        "python": sys.version,
        "scope": (
            "Host-only segmentation; no neural encoder/controller, teacher, output or action calls."
        ),
    }
    try:
        training, authored = training_inputs(), fixtures()
        report["training_inputs_sha256"] = canonical(training)
        report["authored_inputs_sha256"] = canonical(authored)
        all_variants, first_pass = [], {"document": [], "query": []}
        with (args.output / "records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in training + authored:
                guard()
                result = qualify_input(row)
                stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
                all_variants.extend(result["variants"])
                if row["kind"] != "fixture":
                    first_pass[row["kind"]].append(result["variants"][0])
        report["input_records"] = {"training": len(training), "authored": len(authored)}
        report["counts"] = count_invocations(all_variants)
        report["training_one_shot"] = {
            kind: {
                **count_invocations(rows),
                "raw_event_count_distribution": dict(
                    sorted(Counter(v["raw_events"] for v in rows).items())
                ),
                "raw_byte_count_distribution": dict(
                    sorted(Counter(v["raw_bytes"] for v in rows).items())
                ),
            }
            for kind, rows in first_pass.items()
        }
        report["actual_model_work"] = dict.fromkeys(
            (
                "encoder_steps",
                "writer_steps",
                "query_steps",
                "teacher_offers",
                "output_decisions",
                "actions",
            ),
            0,
        )
        documents = report["training_one_shot"]["document"]
        report["preservation_passed"] = True
        report["document_event_reduction"] = (
            documents["conceptual_controller_events"] < documents["conceptual_byte_baseline_steps"]
        )
        report["qualified_for_encoder_integration_qa"] = report["document_event_reduction"]
        report["status"] = "completed"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        try:
            guard()
            report["source_sha256_after"] = sources()
            assert report["source_sha256_after"] == manifest
            assert all(file_hash(ROOT / name) == expected for name, expected in INPUTS.items())
            report["forbidden_module_names"] = sorted(
                name
                for name in sys.modules
                if name == "torch"
                or name.startswith(("torch.", "numpy", "humanbrain._native", "humanbrain.native"))
            )
            assert not report["forbidden_module_names"]
            report["imported_humanbrain_modules"] = sorted(
                n for n in sys.modules if n.startswith("humanbrain")
            )
            assert report["imported_humanbrain_modules"] == ["humanbrain", "humanbrain.byte_events"]
        except BaseException:
            report["status"] = "failed"
            report["finalization_error"] = traceback.format_exc()
        report["elapsed_seconds"] = time.perf_counter() - START
        write_json(args.output / "report.json", report)
        artifacts = (
            {"records.jsonl": file_hash(args.output / "records.jsonl")}
            if (args.output / "records.jsonl").exists()
            else {}
        )
        write_json(
            args.output / "completion.json",
            {
                "status": "passed" if report["status"] == "completed" else "failed",
                "qualified_for_encoder_integration_qa": report.get(
                    "qualified_for_encoder_integration_qa", False
                ),
                "report_sha256": file_hash(args.output / "report.json"),
                "artifacts": artifacts,
                "source_revision": args.revision,
                "admission_sha256": args.admission_sha,
                "elapsed_seconds": time.perf_counter() - START,
            },
        )
    return report["status"] == "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run",))
    for name in ("output", "admission"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("admission-sha", "revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--deadline-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_DEADLINE_UNIX")
    )
    parser.add_argument(
        "--launch-started-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_START_UNIX")
    )
    args = parser.parse_args()
    passed = run(args)
    print(
        json.dumps(
            {
                "status": "passed" if passed else "failed",
                "completion_sha256": file_hash(args.output / "completion.json"),
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
