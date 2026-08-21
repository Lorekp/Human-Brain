from collections import Counter
import json
from pathlib import Path

from benchmarks.language_gate import _baseline_counts, _train_baselines, partition_files


def test_train_only_baselines_do_not_join_files() -> None:
    result = _baseline_counts(
        _train_baselines(["ab", "ca"], max_order=3),
        "abca",
        predictions=3,
        max_order=3,
    )
    assert result["unigram"]["token"] == ord("a")
    # There is no learned b->c edge across the two training-file boundary.
    assert result["bigram"]["correct"] == 2


def test_path_hash_partitions_are_disjoint_and_stable(tmp_path: Path) -> None:
    for index in range(100):
        (tmp_path / f"module_{index}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
    arguments = {
        "bucket_count": 10,
        "train_buckets": range(7),
        "validation_buckets": (7, 8),
        "test_buckets": (9,),
    }
    first = partition_files(tmp_path, **arguments)
    second = partition_files(tmp_path, **arguments)
    assert first == second
    paths = [path for split in first.values() for path in split]
    assert len(paths) == len(set(paths)) == 100


def test_baseline_tie_break_is_deterministic() -> None:
    result = _baseline_counts(
        _train_baselines(["ba"], max_order=2),
        "bbb",
        predictions=2,
        max_order=2,
    )
    assert result["unigram"]["token"] == min(Counter(b"ba"))


def test_post_test_protocol_locks_consumed_test_split() -> None:
    protocol_path = Path(__file__).parents[1] / "benchmarks" / "code_language_protocol_v4.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["test_locked"] is True
