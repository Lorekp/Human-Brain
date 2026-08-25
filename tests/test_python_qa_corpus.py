"""Red tests for the Python-source-text QA corpus (``humanbrain.python_qa_corpus``).

Mirrors tests/test_qa_corpus.py for the coding-transfer corpus:

- build determinism (two in-process builds agree byte-for-byte; digest stable);
- budget: ~100 triples, per-family coverage in BOTH cohorts;
- BY-FILE split disjointness: no source file (and no anchor) appears in both
  cohorts; roles agree with entry cohort labels;
- answer-derivability for every committed artifact entry;
- artifact/manifest hash integrity, with a staleness escape hatch when the
  tree has moved past the pinned git head;
- seed-keyed curriculum over TRAIN only, mirroring qa_corpus.build_curriculum;
- qa_prompt framing byte-stability shared with the Bible corpus;
- tampered entries are rejected.
"""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from humanbrain import python_qa_corpus as pqc
from humanbrain.qa_corpus import QA_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "experiments" / "python_qa_corpus" / "python_qa_corpus_v1.json"
MANIFEST_PATH = ROOT / "experiments" / "python_qa_corpus" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not ARTIFACT_PATH.exists() or not MANIFEST_PATH.exists(),
    reason="python_qa_corpus artifacts not built yet (run tools/build_python_qa_corpus.py)",
)


# ---------------------------------------------------------------------------
# Build determinism
# ---------------------------------------------------------------------------


def test_double_build_is_byte_deterministic_and_digest_stable() -> None:
    first = pqc.canonical_json(pqc.build_corpus())
    second = pqc.canonical_json(pqc.build_corpus())
    assert first == second
    assert pqc.corpus_digest() == sha256(
        pqc.canonical_json(pqc.corpus_record())
    ).hexdigest()
    assert len(pqc.corpus_digest()) == 64


def test_split_roles_are_deterministic_and_floor_guarded() -> None:
    relpaths = pqc.discover_source_files()
    roles_a = pqc.split_file_roles(relpaths)
    roles_b = pqc.split_file_roles(relpaths)
    assert roles_a == roles_b
    val = [path for path, role in roles_a.items() if role == "VAL"]
    train = [path for path, role in roles_a.items() if role == "TRAIN"]
    assert len(val) >= pqc.MIN_VAL_FILES
    assert len(train) >= pqc.MIN_TRAIN_FILES
    assert set(val) | set(train) == set(relpaths)
    assert not set(val) & set(train)


def test_pinned_module_pool_is_the_humanbrain_package_top_level() -> None:
    relpaths = pqc.discover_source_files()
    assert 25 <= len(relpaths) <= 40  # ~30 files
    assert "__init__.py" not in relpaths
    assert all(path.endswith(".py") for path in relpaths)


# ---------------------------------------------------------------------------
# Budget and family coverage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_budget_hits_target_with_family_coverage_in_both_cohorts(artifact: dict) -> None:
    counts = artifact["counts"]
    total = counts["TRAIN"]["total"] + counts["VAL"]["total"]
    assert 90 <= total <= 120, f"corpus budget drifted: {total}"
    assert counts["VAL"]["total"] >= 12
    for cohort in ("TRAIN", "VAL"):
        for family in pqc.FAMILIES:
            assert counts[cohort][family] >= 4, (
                f"{cohort}/{family} under-represented: {counts[cohort][family]}"
            )
        assert counts[cohort]["total"] == sum(
            counts[cohort][family] for family in pqc.FAMILIES
        )


def test_entries_have_dense_ordinals_and_unique_ids(artifact: dict) -> None:
    ordinals: list[int] = []
    ids: list[str] = []
    for cohort in ("TRAIN", "VAL"):
        for record in artifact["entries"][cohort]:
            ordinals.append(record["ordinal"])
            ids.append(record["qa_id"])
            assert record["cohort"] == cohort
    assert sorted(ordinals) == list(range(len(ordinals)))
    assert len(set(ids)) == len(ids)
    assert all(qa_id.startswith("_") or qa_id.isidentifier() for qa_id in ids)


# ---------------------------------------------------------------------------
# Split-disjointness (BY FILE — the core red-test requirement)
# ---------------------------------------------------------------------------


def test_no_source_file_or_anchor_appears_in_both_cohorts(artifact: dict) -> None:
    train_files = {e["source_file"] for e in artifact["entries"]["TRAIN"]}
    val_files = {e["source_file"] for e in artifact["entries"]["VAL"]}
    assert not train_files & val_files, f"file leak across cohorts: {train_files & val_files}"

    train_anchors = {
        e["references"][0]
        for e in artifact["entries"]["TRAIN"]
    }
    val_anchors = {e["references"][0] for e in artifact["entries"]["VAL"]}
    assert not train_anchors & val_anchors

    # Roles recorded at build time must agree with where each entry landed.
    roles = artifact["roles"]
    for cohort in ("TRAIN", "VAL"):
        for record in artifact["entries"][cohort]:
            assert roles[record["source_file"]] == cohort


def test_every_entry_target_is_public(artifact: dict) -> None:
    for cohort in ("TRAIN", "VAL"):
        for record in artifact["entries"][cohort]:
            leaf = record["target"].split(".")[-1]
            assert not leaf.startswith("_"), record["references"]


# ---------------------------------------------------------------------------
# Answer derivability + framing
# ---------------------------------------------------------------------------


def test_every_artifact_answer_is_derivable_from_context(artifact: dict) -> None:
    total = 0
    for cohort in ("TRAIN", "VAL"):
        entries = artifact["entries"][cohort]
        for record in entries:
            total += 1
            context_text = " ".join(record["context"]).casefold()
            assert record["answer"].strip()
            assert record["answer"].casefold() in context_text, (
                f"{record['references'][0]}: answer not derivable from its context lines"
            )
            assert record["question"].endswith("?")
            assert record["family"] in pqc.FAMILIES
    assert total >= 90


def test_questions_are_unambiguous_within_a_cohort(artifact: dict) -> None:
    for cohort in ("TRAIN", "VAL"):
        questions = [record["question"] for record in artifact["entries"][cohort]]
        assert len(set(questions)) == len(questions), f"duplicate question in {cohort}"


def test_prompt_framing_matches_bible_qa_machinery(artifact: dict) -> None:
    record = artifact["entries"]["VAL"][0]
    prompt = pqc.prompt_for_entry(record)
    expected_tail = f"Question: {record['question']}\nanswer:"
    assert prompt.endswith("answer:")
    assert prompt.encode("utf-8").endswith(expected_tail.encode("utf-8"))
    assert prompt.startswith(record["context"][0])
    growth = pqc.growth_text_for_entry(record)
    assert growth == prompt + " " + record["answer"]

    from humanbrain import qa_corpus

    # Identical framing function as the Bible corpus (single framing seam).
    assert prompt == qa_corpus.qa_prompt(tuple(record["context"]), record["question"])
    curriculum_task = pqc.build_curriculum(0)["train_tasks"][0]
    framed = qa_corpus.prompt_for_task(curriculum_task)
    assert framed.endswith("answer:") and QA_SCHEMA_VERSION.startswith("humanbrain/")


# ---------------------------------------------------------------------------
# Artifact / manifest integrity
# ---------------------------------------------------------------------------


def _current_git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_manifest_hashes_match_artifact_bytes_and_canonical_recompute(artifact: dict) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact_bytes = ARTIFACT_PATH.read_bytes()
    assert manifest["artifact_sha256"] == sha256(artifact_bytes).hexdigest()
    recomputed = sha256(
        pqc.canonical_json({"entries": artifact["entries"]})
    ).hexdigest()
    assert artifact["corpus_sha256"] == manifest["corpus_sha256"]
    # The library's live digest pins to the same content while sources are
    # unchanged; once HEAD moves past the pinned commit this is expected drift.
    if manifest.get("git_head") == _current_git_head():
        assert pqc.corpus_digest() == manifest["corpus_sha256"]
        for file_row in artifact["files"]:
            source = ROOT / "src" / "humanbrain" / file_row["path"]
            assert source.exists(), file_row["path"]
            assert (
                sha256(source.read_bytes()).hexdigest() == file_row["sha256"]
            ), f"stale corpus: {file_row['path']} changed since the pinned head"


# ---------------------------------------------------------------------------
# Seed-keyed curriculum over TRAIN
# ---------------------------------------------------------------------------


def test_curriculum_is_seed_keyed_and_train_only(artifact: dict) -> None:
    a3 = pqc.build_curriculum(3)
    b3 = pqc.build_curriculum(3)
    c4 = pqc.build_curriculum(4)
    assert a3["sha256"] == b3["sha256"]
    assert a3["sha256"] != c4["sha256"]
    assert "seed/3" in str(a3["identifier_domain"])
    assert "seed/3" in str(a3["order_domain"])

    train_pairs = {
        (record["question"], record["answer"])
        for record in artifact["entries"]["TRAIN"]
    }
    val_questions = {
        record["question"] for record in artifact["entries"]["VAL"]
    }
    tasks = tuple(a3["train_tasks"]) + tuple(a3["calibration_tasks"])
    ordinals = sorted(task.ordinal for task in tasks)
    assert ordinals == list(range(len(tasks)))
    assert len({task.qa_id for task in tasks}) == len(tasks)

    for task in tasks:
        assert (task.question, task.answer) in train_pairs, task.qa_id
        assert task.question not in val_questions

    with pytest.raises(ValueError):
        pqc.build_curriculum(0, calibration_task_count=0)
    with pytest.raises(ValueError):
        pqc.build_curriculum("7")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tamper rejection
# ---------------------------------------------------------------------------


def test_validator_rejects_undervivable_tampered_entry(artifact: dict) -> None:
    record = dict(artifact["entries"]["VAL"][0])
    pristine = dict(record)
    record["answer"] = "definitely not present anywhere in this context"
    with pytest.raises(ValueError):
        pqc._validate_entry_record(record)

    record = dict(pristine)
    record["family"] = "made_up_family"
    with pytest.raises(ValueError):
        pqc._validate_entry_record(record)

    record = dict(pristine)
    record["context"] = []
    with pytest.raises(ValueError):
        pqc._validate_entry_record(record)


def test_candidate_validation_rejects_answer_not_in_context() -> None:
    candidate = pqc.Candidate(
        family="function_summary",
        source_file="zzz.py",
        module="humanbrain.zzz",
        target="ghost",
        anchor="zzz.py::ghost::function_summary",
        context=("# module humanbrain.zzz (zzz.py)", '"""Placeholder text."""'),
        question="In the module humanbrain.zzz, what does the function ghost do?",
        answer="something never stated",
    )
    with pytest.raises(ValueError):
        pqc.validate_candidate(candidate)
