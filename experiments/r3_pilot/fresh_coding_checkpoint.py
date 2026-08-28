"""Freeze the task-independent CellLanguageModel for the fresh coding cohort.

This stage intentionally runs before ``fresh_coding_challenge.py`` exists.  It
learns only from the adjacent generic corpus, writes one immutable cell/synapse
checkpoint, and records enough hashes to make the freeze independently
auditable.  No target task, target solution, evaluator, or external neural
model is imported here.
"""

from __future__ import annotations

import ast
import json
import platform
import subprocess
import sys
import tempfile
import zlib
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from humanbrain.cortex import HumanCellParameters
from humanbrain.language import CellLanguageModel, ModelConfig


FORMAT = "humanbrain.fresh-coding-checkpoint.v1"
CORPUS_FORMAT = "humanbrain.fresh-coding-generic-corpus.v1"
CORPUS_PATH = Path(__file__).with_name("fresh_coding_generic_corpus.json")
CHECKPOINT_PATH = Path(__file__).with_name("fresh_coding_checkpoint.npz")
MANIFEST_PATH = Path(__file__).with_name("fresh_coding_checkpoint.manifest.json")
FORBIDDEN_CHALLENGE_PATH = Path(__file__).with_name("fresh_coding_challenge.py")

TASK_OPEN = "<|task|>"
SOURCE_OPEN = "<|source|>"
SOURCE_END = "# <END>"
CANDIDATE_OPEN = "<|candidate|>"
FEEDBACK_OPEN = "<|feedback|>"
REVISION_OPEN = "<|revision|>"
ALLOWED_PARAMETERS = ("arg0", "arg1", "arg2")

FROZEN_RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.11.15",
    "numpy_version": "2.4.6",
}

MODEL_CONFIG = ModelConfig(
    max_order=64,
    max_phrases=512,
    min_phrase_frequency=2,
    random_seed=7,
    cell_parameters=HumanCellParameters(),
)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _assert_frozen_runtime() -> None:
    actual = _runtime_identity()
    if actual != FROZEN_RUNTIME:
        raise RuntimeError(f"frozen runtime mismatch: expected {FROZEN_RUNTIME}, got {actual}")


def _validate_source(source: str, signature: str) -> None:
    if any(marker in source for marker in (TASK_OPEN, SOURCE_OPEN, SOURCE_END, FEEDBACK_OPEN)):
        raise ValueError("source contains a protocol delimiter")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != "solve":
        raise ValueError("each source must contain exactly one top-level solve function")
    args = functions[0].args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults:
        raise ValueError("solve must use only required positional parameters")
    names = tuple(arg.arg for arg in args.args)
    if names != ALLOWED_PARAMETERS:
        raise ValueError(f"solve parameters must be exactly {ALLOWED_PARAMETERS}")
    if signature != f"solve({', '.join(names)})":
        raise ValueError("declared signature does not match source")


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != CORPUS_FORMAT:
        raise ValueError(f"unexpected corpus format: {raw.get('format')!r}")
    protocol = raw.get("protocol")
    expected_protocol = {
        "entrypoint": "solve",
        "parameters": list(ALLOWED_PARAMETERS),
        "task_open": TASK_OPEN,
        "source_open": SOURCE_OPEN,
        "source_end": SOURCE_END,
        "candidate_open": CANDIDATE_OPEN,
        "feedback_open": FEEDBACK_OPEN,
        "revision_open": REVISION_OPEN,
    }
    if protocol != expected_protocol:
        raise ValueError("corpus protocol drifted from the frozen builder")
    examples = raw.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("generic corpus needs at least one example")
    seen_ids: set[str] = set()
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("corpus examples must be objects")
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id or example_id in seen_ids:
            raise ValueError("corpus example ids must be unique non-empty strings")
        seen_ids.add(example_id)
        task = example.get("task")
        signature = example.get("signature")
        feedback_json = example.get("feedback_json")
        if not isinstance(task, str) or not task.strip() or "<|" in task:
            raise ValueError(f"invalid task text in {example_id}")
        if not isinstance(signature, str) or not isinstance(feedback_json, str):
            raise ValueError(f"missing signature or raw feedback in {example_id}")
        parsed_feedback = json.loads(feedback_json)
        if not isinstance(parsed_feedback, dict) or _canonical_json(parsed_feedback) != feedback_json:
            raise ValueError(f"feedback_json must be a canonical JSON object in {example_id}")
        if set(parsed_feedback) != {"cases", "passed"} or parsed_feedback["passed"] is not False:
            raise ValueError(f"feedback_json must be a failing visible-case wrapper in {example_id}")
        cases = parsed_feedback["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"feedback_json cases must be a non-empty list in {example_id}")
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError(f"feedback cases must be objects in {example_id}")
            required = {"args", "expected", "passed"}
            observation = {"actual", "exception"}.intersection(case)
            if (
                not required.issubset(case)
                or len(observation) != 1
                or set(case) != required | observation
                or not isinstance(case["args"], list)
                or len(case["args"]) != len(ALLOWED_PARAMETERS)
                or case["passed"] is not False
            ):
                raise ValueError(f"invalid raw visible case in {example_id}")
        for field in ("initial_source", "revision_source"):
            source = example.get(field)
            if not isinstance(source, str) or not source.endswith("\n"):
                raise ValueError(f"{field} must be newline-terminated in {example_id}")
            _validate_source(source, signature)
    return raw


def _task_prompt(example: dict[str, Any]) -> str:
    return (
        f"{TASK_OPEN}\nReturn only Python source.\n"
        f"signature: {example['signature']}\n"
        f"task: {example['task'].strip()}\n"
    )


def training_texts(corpus: dict[str, Any]) -> tuple[str, ...]:
    """Return direct-solve and raw-feedback repair dialogues in stable order."""

    texts: list[str] = []
    for example in corpus["examples"]:
        prompt = _task_prompt(example)
        correct = example["revision_source"]
        texts.append(f"{prompt}{SOURCE_OPEN}\n{correct}{SOURCE_END}\n")
        texts.append(
            f"{prompt}{CANDIDATE_OPEN}\n{example['initial_source']}{SOURCE_END}\n"
            f"{FEEDBACK_OPEN}\n{example['feedback_json']}\n"
            f"{REVISION_OPEN}\n{correct}{SOURCE_END}\n"
        )
    return tuple(texts)


def _training_digest(texts: tuple[str, ...]) -> str:
    digest = sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _require_tracked_clean_sources(paths: tuple[Path, ...]) -> str:
    relative_paths = [_repo_relative(path) for path in paths]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative_paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError("checkpoint source and corpus must be tracked before freeze")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise RuntimeError(f"checkpoint source or corpus is dirty:\n{status.rstrip()}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def build_checkpoint(
    *,
    corpus_path: Path = CORPUS_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    forbidden_challenge_path: Path = FORBIDDEN_CHALLENGE_PATH,
) -> dict[str, Any]:
    """Create the generic checkpoint and manifest exactly once."""

    _assert_frozen_runtime()
    if forbidden_challenge_path.exists():
        raise FileExistsError(
            f"challenge already exists; generic checkpoint must be frozen first: {forbidden_challenge_path}"
        )
    for output in (checkpoint_path, manifest_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen output: {output}")
    freeze_source_commit = _require_tracked_clean_sources((Path(__file__), corpus_path))

    corpus = load_corpus(corpus_path)
    texts = training_texts(corpus)
    model = CellLanguageModel.grow(texts, config=MODEL_CONFIG)
    expected_training_digest = _training_digest(texts)
    if model.training_digest != expected_training_digest:
        raise RuntimeError("CellLanguageModel training digest does not match serialized dialogues")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    checkpoint_created = False
    try:
        with tempfile.NamedTemporaryFile(
            prefix="fresh-coding-",
            suffix=".npz",
            dir=checkpoint_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        checkpoint_sha256 = model.save(temporary_path)
        checkpoint_bytes = temporary_path.read_bytes()
        if sha256(checkpoint_bytes).hexdigest() != checkpoint_sha256:
            raise RuntimeError("checkpoint hash changed after save")
        with checkpoint_path.open("xb") as stream:
            stream.write(checkpoint_bytes)
        checkpoint_created = True

        manifest: dict[str, Any] = {
            "format": FORMAT,
            "freeze_source_commit": freeze_source_commit,
            "source": {
                "path": _repo_relative(Path(__file__)),
                "sha256": _sha256_file(Path(__file__)),
            },
            "corpus": {
                "path": _repo_relative(corpus_path),
                "sha256": _sha256_file(corpus_path),
                "format": CORPUS_FORMAT,
                "example_count": len(corpus["examples"]),
                "training_text_count": len(texts),
                "serialized_training_sha256": expected_training_digest,
            },
            "checkpoint": {
                "path": _repo_relative(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "model": {
                "class": "humanbrain.language.CellLanguageModel",
                "config": asdict(MODEL_CONFIG),
                "training_digest": model.training_digest,
                "vocabulary_size": model.vocabulary_size,
                "grown_branches": model.grown_branches,
                "learned_inference": "simulated_cells_and_synapses_only",
                "external_neural_models": [],
            },
            "protocol": corpus["protocol"],
            "scope": {
                "claim": "scaffolded_body_synthesis_only",
                "corpus_policy": "explicit_allowlist_only",
                "forbidden_challenge_path": _repo_relative(forbidden_challenge_path),
                "target_tasks_present_at_freeze": False,
            },
            "runtime": {
                **_runtime_identity(),
                "byteorder": sys.byteorder,
                "platform_machine": platform.machine(),
                "platform_system": platform.system(),
                "zlib_compile_version": zlib.ZLIB_VERSION,
                "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            },
        }
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(manifest_text)
        return manifest
    except BaseException:
        if checkpoint_created and not manifest_path.exists():
            checkpoint_path.unlink(missing_ok=True)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("this one-shot freeze builder accepts no options")
    manifest = build_checkpoint()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
