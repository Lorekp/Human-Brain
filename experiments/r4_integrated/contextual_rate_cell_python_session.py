"""One declared rate-cell acquisition preflight in the executed Python loop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.r4_integrated.contextual_full_choice_python_session import (
    FIXTURE,
    ROOT,
    _item,
    _summary,
    acquisition_gate,
    load_fixture,
)
from humanbrain.rate_cell_reader import RateCellReader, SpanInventory, TokenDocument, TokenInput

PROTOCOL = Path(__file__).with_name("contextual_rate_cell_protocol.md")
ARMS = ("trainable", "frozen_producer", "bypass", "frozen", "external")
PHASES = (
    "actual_before_growth",
    "alternate_before_growth",
    "actual_after_growth",
    "alternate_after_growth",
)
ACQUISITION_SECONDS = 1800
RSS_MIB = 2048
PAYLOAD_BYTES = 1024**3


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return sha256(canonical(value).encode()).hexdigest()


def retained_size(*roots) -> int:
    """Count Python/numpy owned buffers once, following view bases and shared inputs."""
    total, seen, pending = 0, set(), list(roots)
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            if value.base is not None:
                pending.append(value.base)
        elif isinstance(value, Mapping):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif is_dataclass(value):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif type(value).__module__.startswith(("humanbrain.", "experiments.r4_integrated.")):
            if hasattr(value, "__dict__"):
                pending.append(vars(value))
            pending.extend(
                getattr(value, name)
                for name in getattr(type(value), "__slots__", ())
                if name not in ("__dict__", "__weakref__")
            )
    return total


@dataclass(frozen=True)
class Decision:
    candidate_index: int | None
    emitted: bytes | None
    query: bytes
    query_input_sha256: str
    checkpoint_sha256: str
    storage_sha256: str
    mode: str
    stop_reason: str | None
    p_stop: float
    stop_logit: float
    span_scores_sha256: str
    eligibility: tuple = ()
    sha256: str = ""

    @property
    def stopped(self) -> bool:
        return self.candidate_index is None

    def record(self) -> dict:
        return {
            **asdict(self),
            "query": self.query.hex(),
            "emitted": None if self.emitted is None else self.emitted.hex(),
        }


@dataclass(frozen=True)
class Lesson:
    original: Decision
    answer: bytes | None
    teacher_id: str
    attempt_json: str
    teacher_json: str
    sha256: str = ""

    def record(self) -> dict:
        return {
            **asdict(self),
            "original": self.original.record(),
            "answer": None if self.answer is None else self.answer.hex(),
        }


class ExecutedReader:
    """Bind numerical updates to this arm's immutable action and teacher receipts."""

    def __init__(self, mode: str, inventory: SpanInventory, queries: dict[bytes, TokenInput]):
        if any(raw != tokens.raw for raw, tokens in queries.items()):
            raise ValueError("query cache keys must match authenticated input bytes")
        self.mode = mode
        self.core = RateCellReader(mode="frozen" if mode == "external" else mode)
        self.inventory = inventory
        self.queries = MappingProxyType(dict(queries))
        self.training_pool = inventory.sha256
        self.feedback_root = digest([])

    @property
    def checkpoint_sha256(self) -> str:
        return digest(
            {
                "core": self.core.checkpoint_sha256,
                "source": self.inventory.sha256,
                "feedback": self.feedback_root,
                "mode": self.mode,
            }
        )

    def predict(self, query: bytes):
        tokens = self.queries[query]
        method = self.core.external_only if self.mode == "external" else self.core.predict
        return method(tokens, self.inventory)

    def respond(self, query: bytes) -> Decision:
        parent = self.checkpoint_sha256
        prediction = self.predict(query)
        index = prediction.candidate_index
        decision = Decision(
            index,
            None if index is None else self.inventory.candidates[index].emitted,
            query,
            self.queries[query].sha256,
            parent,
            self.inventory.sha256,
            self.mode,
            prediction.stop_reason,
            prediction.p_stop,
            prediction.stop_logit,
            sha256(prediction.span_scores.tobytes()).hexdigest(),
        )
        if self.checkpoint_sha256 != parent:
            raise RuntimeError("prediction mutated persistent state")
        return replace(decision, sha256=digest(decision.record()))

    def _validate(self, lesson: Lesson) -> None:
        from humanbrain.full_choice_evidence_memory import FullChoiceEvidenceMemory

        if self.inventory.sha256 != self.training_pool:
            raise ValueError("learning requires the sealed TRAIN inventory")
        original = lesson.original
        if (
            original.storage_sha256 != self.training_pool
            or original.mode != self.mode
            or original.sha256 != digest(replace(original, sha256="").record())
            or lesson.sha256 != digest(replace(lesson, sha256="").record())
        ):
            raise ValueError("lesson, own decision or source custody changed")
        index = original.candidate_index
        if index is None:
            valid = original.emitted is None
        else:
            valid = (
                type(index) is int
                and 0 <= index < len(self.inventory.candidates)
                and original.emitted == self.inventory.candidates[index].emitted
            )
        if (
            not valid
            or original.query not in self.queries
            or original.query_input_sha256 != self.queries[original.query].sha256
        ):
            raise ValueError("original selected occurrence or query changed")
        attempt, teacher = json.loads(lesson.attempt_json), json.loads(lesson.teacher_json)
        FullChoiceEvidenceMemory._validate_teacher(lesson.answer, attempt, teacher)
        if attempt.get("own_decision_sha256") != original.sha256:
            raise ValueError("execution receipt is not bound to this arm's decision")
        if lesson.answer is not None and not self.inventory.gold_indices(lesson.answer):
            raise ValueError("authorized whole answer is absent from TRAIN inventory")

    def teacher_lesson(
        self,
        original: Decision,
        answer: bytes | None,
        teacher_id: str,
        attempt: dict,
        teacher: dict,
    ) -> Lesson:
        if original.checkpoint_sha256 != self.checkpoint_sha256 or not teacher_id:
            raise ValueError("new lesson requires this arm's current first attempt")
        lesson = Lesson(original, answer, teacher_id, canonical(attempt), canonical(teacher))
        lesson = replace(lesson, sha256=digest(lesson.record()))
        self._validate(lesson)
        return lesson

    def observe(self, lesson: Lesson) -> dict:
        self._validate(lesson)
        before = self.checkpoint_sha256
        gold = None if lesson.answer is None else self.inventory.gold_indices(lesson.answer)
        receipt = self.core.observe(self.queries[lesson.original.query], self.inventory, gold)
        receipt.update(
            {
                "lesson_sha256": lesson.sha256,
                "own_decision_sha256": lesson.original.sha256,
                "source_pool_sha256": self.training_pool,
                "adapter_before": before,
                "teacher_id": lesson.teacher_id,
                "gold_indices": gold,
                "new_execution_credit": False,
            }
        )
        self.feedback_root = digest({"parent": self.feedback_root, "offer": receipt})
        receipt["adapter_after"] = self.checkpoint_sha256
        return receipt

    def snapshot(self) -> dict:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "core": self.core.snapshot(),
            "inventory_sha256": self.inventory.sha256,
            "feedback_root": self.feedback_root,
            "candidate_count": len(self.inventory.candidates),
        }


def gate(phases: dict) -> dict:
    renamed = {phase: {"update": result["trainable"]} for phase, result in phases.items()}
    result = acquisition_gate(renamed)
    stops = {
        phase: row["trainable"]["summary"]["correct_stop"] >= 15 for phase, row in phases.items()
    }
    contribution = {}
    for phase in ("actual_before_growth", "actual_after_growth"):
        candidate = phases[phase]["trainable"]["summary"]
        for control in ("frozen_producer", "bypass"):
            reference = phases[phase][control]["summary"]
            contribution[f"{phase}:{control}"] = {
                "supported_advantage": candidate["executed_success"]
                - reference["executed_success"],
                "stop_advantage": candidate["correct_stop"] - reference["correct_stop"],
            }
    result["all_phase_stop_thresholds"] = stops
    result["learned_cell_contribution"] = contribution
    result["passed"] = (
        result["passed"]
        and all(stops.values())
        and all(
            row["supported_advantage"] >= 8 and row["stop_advantage"] >= 0
            for row in contribution.values()
        )
    )
    return result


def run(output: Path, model_dir: Path, numerical_receipt: Path) -> dict:
    output = output.resolve()
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("prior artifacts are immutable; choose an unused output path")
    if (
        os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
        or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve() != ROOT / ".native-cache"
    ):
        raise RuntimeError("set CPU and checkout native-cache before Python starts")
    if any(
        os.environ.get(name) != "1"
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("declare single-thread numerical runtimes before Python starts")
    from benchmarks.native_scaling import _process_memory
    from experiments.r4_integrated import contextual_evidence_python_session as shared
    from experiments.r4_integrated.contextual_pairwise_python_session import _motor
    from humanbrain.frozen_context_encoder import FrozenContextEncoder
    from humanbrain.native import loaded_build_info

    preflight = json.loads(numerical_receipt.read_bytes())
    if preflight.get("status") != "passed":
        raise ValueError("a passed held-source numerical/cost receipt is required")
    remaining = 2700 - (time.time() - preflight["started_unix"])
    if preflight["elapsed_seconds"] > 900 or remaining <= 0:
        raise ValueError("the declared numerical/overall preflight budget is exhausted")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if preflight["source_revision"] != revision:
        raise ValueError("numerical preflight does not bind this source revision")
    paths = tuple(
        dict.fromkeys(
            (
                *shared.SOURCE_PATHS,
                "experiments/r4_integrated/contextual_pairwise_python_session.py",
                "experiments/r4_integrated/contextual_full_choice_python_session.py",
                "src/humanbrain/full_choice_evidence_memory.py",
                "src/humanbrain/rate_cell_reader.py",
                "experiments/r4_integrated/contextual_rate_cell_preflight.py",
                Path(__file__).relative_to(ROOT).as_posix(),
                PROTOCOL.relative_to(ROOT).as_posix(),
                FIXTURE.relative_to(ROOT).as_posix(),
            )
        )
    )
    sources = {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    if any(sources.get(p) != h for p, h in preflight["source_sha256"].items()):
        raise ValueError("held numerical source changed before acquisition")
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".runtime").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextual-rate-cell-", dir=ROOT / ".runtime"))
    started = time.perf_counter()
    deadline = started + min(ACQUISITION_SECONDS, remaining)
    readers, arrays = {}, {}
    report = {
        "terminal": "running",
        "source_revision": revision,
        "source_sha256": sources,
        "numerical_receipt": {
            "path": str(numerical_receipt),
            "sha256": sha256(numerical_receipt.read_bytes()).hexdigest(),
        },
        "workspace_root": str(work),
        "training": [],
        "phases": {},
        "events": [],
        "scope": "exposed TRAIN acquisition; retained raw sources and frozen MiniLM words",
        "control_scope": "evidence learner controls share an already learned neuronal edit motor",
        "limits": {
            "seconds": ACQUISITION_SECONDS,
            "parent_rss_mib": RSS_MIB,
            "retained_payload_bytes": PAYLOAD_BYTES,
        },
    }

    def event(name, **fields):
        row = {"event": name, "elapsed_seconds": time.perf_counter() - started, **fields}
        report["events"].append(row)
        with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(row) + "\n")
        print(canonical(row), flush=True)

    def check_budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError("declared acquisition time budget exhausted")
        if _process_memory()["working_set_mib"] > RSS_MIB:
            raise MemoryError("declared parent resident memory limit exceeded")

    def checkpoint(prefix):
        for arm, reader in readers.items():
            arrays.update(
                {f"{prefix}_{arm}_{k}": v for k, v in reader.core.checkpoint_arrays().items()}
            )

    try:
        fixture = load_fixture()
        tasks, queries = fixture["tasks"], {}
        by_id = {task["id"]: task for task in tasks}
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to(ROOT / ".native-cache"):
            raise RuntimeError("motor loaded a native library outside this checkout cache")
        report["native_motor"] = {
            "source_digest": native.source_digest,
            "library": str(native.library_path),
            "binary_sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        motor = _motor(work, report)
        began = time.perf_counter()
        encoder = FrozenContextEncoder(model_dir, max_tokens=256, threads=1)
        report["encoder"] = encoder.metadata
        token_records = []

        def encode(raw: bytes, kind: str) -> TokenInput:
            check_budget()
            encoded = encoder.encode(raw)
            tokens = TokenInput.from_encoding(raw, encoded)
            name = f"input_{len(token_records):03d}"
            arrays[name] = tokens.vectors
            token_records.append(
                {
                    "kind": kind,
                    "array": name,
                    **tokens.record(),
                    "subtoken_count": len(encoded.token_features),
                    "encoder_token_count": encoded.token_count,
                }
            )
            return tokens

        documents = tuple(
            TokenDocument(
                doc["id"].encode(),
                b"rate-cell-public-train-v1",
                encode(doc["text"].encode(), "TRAIN_document"),
            )
            for doc in fixture["documents"]
        )
        # Frozen preprocessing cache is outside model state. Queries enter the
        # learner only at the scheduled attempt/offer, with no expected answer.
        for task in tasks:
            for query in (task["query"], fixture["alternate_queries"][task["id"]]):
                raw = query.encode()
                if raw not in queries:
                    queries[raw] = encode(raw, "query")
        inventory = SpanInventory(documents, window_words=48)
        if len(inventory.candidates) != 8162:
            raise ValueError("declared TRAIN candidate inventory changed")
        if any(
            task["answer"] is not None and len(inventory.gold_indices(task["answer"].encode())) != 1
            for task in tasks
        ):
            raise ValueError("TRAIN whole-answer custody changed")
        report["input_records"] = token_records
        report["training_inventory"] = {
            "sha256": inventory.sha256,
            "candidates": inventory.candidate_manifest(),
            "document_ids": [d.document_id.hex() for d in documents],
        }
        report["input_preparation_seconds"] = time.perf_counter() - began
        readers = {arm: ExecutedReader(arm, inventory, queries) for arm in ARMS}
        seed = readers["trainable"].core.checkpoint_arrays()
        for reader in readers.values():
            current = reader.core.checkpoint_arrays()
            if any(not np.array_equal(value, current[key]) for key, value in seed.items()):
                raise RuntimeError("control initialization differs")
        checkpoint("initial")
        report["initial"] = {arm: reader.snapshot() for arm, reader in readers.items()}

        def attempt(arm, task, query, folder, diagnostics=False):
            check_budget()
            reader = readers[arm]
            row, decision = shared.attempt(reader, "respond", motor, _item(task, query), folder)
            row["source_storage"] = "retained raw sources and frozen normalized word vectors"
            if diagnostics:
                prediction = reader.predict(query.encode())
                gold = (
                    ()
                    if task["answer"] is None
                    else reader.inventory.gold_indices(task["answer"].encode())
                )
                scores = prediction.span_scores
                row["diagnostics"] = {
                    "p_stop": prediction.p_stop,
                    "stop_logit": prediction.stop_logit,
                    "gold_indices": gold,
                    "span_count": len(scores),
                    "span_scores_sha256": sha256(scores.tobytes()).hexdigest(),
                }
                if gold:
                    best = float(scores[list(gold)].max())
                    wrong = np.ones(len(scores), dtype=bool)
                    wrong[list(gold)] = False
                    row["diagnostics"].update(
                        best_gold_rank=int(1 + np.count_nonzero(scores > best)),
                        gold_to_worst_gap=best - float(scores[wrong].max()),
                    )
            return row, decision

        previous = {}

        def resources():
            # ONNX/native opaque allocations are covered by the measured process
            # memory; model-file bytes and the motor's retained events are separate.
            payload = retained_size(readers, previous, motor)
            encoder_bytes = sum(item["bytes"] for item in report["encoder"]["files"].values())
            motor_bytes = report["motor"]["retained_demonstration_event_bytes"]
            result = {
                "python_numpy_learner_and_motor_bytes": payload,
                "encoder_file_bytes": encoder_bytes,
                "motor_retained_event_bytes": motor_bytes,
                "counted_payload_bytes": payload + encoder_bytes,
                "audit_retained_bytes": retained_size(report, arrays),
                "parent_memory": _process_memory(),
            }
            if result["counted_payload_bytes"] > PAYLOAD_BYTES:
                raise MemoryError("declared retained learner payload limit exceeded")
            return result

        report["resources_after_ingestion"] = resources()
        for step, task_id in enumerate(fixture["schedule"], 1):
            task = by_id[task_id]
            folder = work / "teaching" / f"{step:03d}-{task_id}"
            row = {
                "step": step,
                "pass": (step - 1) // 64 + 1,
                "task": task_id,
                "first": {},
                "lessons": {},
                "new": {},
                "replay": {},
            }
            report["training"].append(row)
            decisions = {}
            for arm in ARMS:
                row["first"][arm], decisions[arm] = attempt(arm, task, task["query"], folder / arm)
            case = shared.Case(
                name=task_id,
                key=task["workspace"]["key"],
                module=task["workspace"]["module"],
                stale=task["workspace"]["stale"],
                value=task["answer"] or task["workspace"]["stale"],
            )
            before = case.files()
            after = case.files(corrected=True) if task["answer"] is not None else before
            teacher = shared.execute(case, after, before, folder / "teacher")
            teacher.update(
                teacher_kind="authorized_exact_answer"
                if task["answer"] is not None
                else "authorized_absent_evidence_stop",
                answer_hex=None if task["answer"] is None else task["answer"].encode().hex(),
                workspace_unchanged=after == before,
            )
            row["teacher"] = teacher
            if not teacher["passed"]:
                raise RuntimeError("shared teacher action/no-action check failed")
            for arm, reader in readers.items():
                check_budget()
                execution = {
                    **row["first"][arm]["execution"],
                    "own_decision_sha256": decisions[arm].sha256,
                }
                lesson = reader.teacher_lesson(
                    decisions[arm],
                    None if task["answer"] is None else task["answer"].encode(),
                    f"{step:03d}:{task_id}",
                    execution,
                    teacher,
                )
                row["lessons"][arm] = lesson.record()
                row["new"][arm] = reader.observe(lesson)
                if arm in previous:
                    row["replay"][arm] = reader.observe(previous[arm])
                previous[arm] = lesson
            event("teaching", step=step, task=task_id, offers=2 * step - 1)
            if step % 64 == 0:
                report.setdefault("resources_by_pass", []).append(resources())
        report["pass_summaries"] = [
            {
                "pass": number,
                **{
                    arm: _summary(
                        [row["first"][arm] for row in report["training"] if row["pass"] == number],
                        [by_id[row["task"]] for row in report["training"] if row["pass"] == number],
                    )
                    for arm in ARMS
                },
            }
            for number in range(1, 5)
        ]
        checkpoint("trained")
        for arm, reader in readers.items():
            changed = sum(
                int(offer[arm]["parameter_update_applied"])
                for row in report["training"]
                for offer in (row["new"], row["replay"])
                if arm in offer
            )
            if (
                reader.core.offer_count != 511
                or reader.core.update_count != changed
                or (arm in ("frozen", "external") and changed != 0)
            ):
                raise RuntimeError("supervised offers or learning-control updates differ")
        learned = {arm: reader.core.checkpoint_arrays() for arm, reader in readers.items()}
        report["trained"] = {arm: reader.snapshot() for arm, reader in readers.items()}

        for phase in PHASES:
            if phase == "actual_after_growth":
                extra = tuple(
                    TokenDocument(
                        doc["id"].encode(),
                        b"rate-cell-growth-only-v1",
                        encode(doc["text"].encode(), "growth_document"),
                    )
                    for doc in fixture["growth_documents"]
                )
                grown = SpanInventory(documents + extra, window_words=48)
                if len(grown.candidates) != 10864:
                    raise ValueError("declared grown candidate inventory changed")
                for reader in readers.values():
                    reader.inventory = grown
                report["grown_inventory"] = {
                    "sha256": grown.sha256,
                    "candidates": grown.candidate_manifest(),
                }
                report["resources_after_growth"] = resources()
            report["phases"][phase] = {}
            for arm in ARMS:
                rows = []
                report["phases"][phase][arm] = {"rows": rows}
                for task in tasks:
                    query = (
                        fixture["alternate_queries"][task["id"]]
                        if phase.startswith("alternate")
                        else task["query"]
                    )
                    row, _ = attempt(
                        arm,
                        task,
                        query,
                        work / phase / arm / task["id"],
                        diagnostics=arm == "trainable",
                    )
                    rows.append(row)
                summary = _summary(rows, tasks)
                report["phases"][phase][arm]["summary"] = summary
                event("evaluation", phase=phase, arm=arm, summary=summary)
            for arm, reader in readers.items():
                current = reader.core.checkpoint_arrays()
                if any(
                    not np.array_equal(value, current[key]) for key, value in learned[arm].items()
                ):
                    raise RuntimeError("evaluation/growth changed learned or optimizer state")
                if (
                    reader.core.checkpoint_sha256
                    != report["trained"][arm]["core"]["checkpoint_sha256"]
                ):
                    raise RuntimeError("evaluation/growth changed learning counters")
        report["acquisition_gate"] = gate(report["phases"])
        report["end"] = {arm: reader.snapshot() for arm, reader in readers.items()}
        report["final_resources"] = resources()
        if sources != {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("source or input fixture changed during the run")
        report["terminal"] = "completed"
    except Exception:
        report["terminal"], report["error"] = "error", traceback.format_exc()
    finally:
        finalization_errors = []
        try:
            checkpoint("final")
            report["parent_memory"] = _process_memory()
            report["source_sha256_after"] = {
                p: sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
            }
            report["source_unchanged"] = sources == report["source_sha256_after"]
            if not report["source_unchanged"]:
                raise RuntimeError("source changed during the attempt")
        except Exception:
            finalization_errors.append(traceback.format_exc())
        try:
            report["audit_array_payload_bytes"] = sum(array.nbytes for array in arrays.values())
            report["array_sha256"] = {
                key: sha256(value.tobytes()).hexdigest() for key, value in arrays.items()
            }
            checkpoint_path = output.with_suffix(".npz")
            with checkpoint_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            report["checkpoint"] = {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
        except Exception:
            finalization_errors.append(traceback.format_exc())
        if finalization_errors:
            report["terminal"] = "error"
            report["finalization_errors"] = finalization_errors
        report["elapsed_seconds"] = time.perf_counter() - started
        report["elapsed_scope"] = (
            "through checkpoint export; final JSON serialization/write excluded"
        )
        if time.perf_counter() >= deadline:
            report["terminal"] = "error"
            report["budget_exhausted"] = True
        try:
            serialized = canonical(report)
        except (TypeError, ValueError):
            # A corrupt/nonfinite reporting value is a failed attempt. Preserve
            # its original evidence as text rather than replacing the first error
            # or leaving an empty file that could look like a terminal receipt.
            report = {
                "terminal": "error",
                "source_revision": revision,
                "source_sha256": sources,
                "workspace_root": str(work),
                "error": report.get("error"),
                "serialization_error": traceback.format_exc(),
                "partial_report_repr": repr(report),
            }
            serialized = canonical(report)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numerical-receipt", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    outcome = run(args.output, args.model_dir, args.numerical_receipt)
    print(
        canonical(
            {
                "terminal": outcome["terminal"],
                "output": str(args.output),
                "gate": outcome.get("acquisition_gate"),
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if outcome["terminal"] == "completed" else 1)
