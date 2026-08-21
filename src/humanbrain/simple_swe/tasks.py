from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
from typing import Callable, Sequence, TypeAlias

from humanbrain.simple_swe.protocol import ProtocolSpec


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
Oracle = Callable[..., JsonValue]


@dataclass(frozen=True)
class HiddenCase:
    args: tuple[JsonValue, ...]
    expected: JsonValue
    distribution: str


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    partition: str
    tier: int
    family: str
    ordinal: int
    entry_point: str
    parameter_names: tuple[str, ...]
    prompt: str
    hidden_test_sha256: str
    task_sha256: str


@dataclass(frozen=True)
class PrivateTask:
    task_id: str
    public_task_sha256: str
    regression_cases: tuple[HiddenCase, ...]
    property_cases: tuple[HiddenCase, ...]
    test_sha256: str


@dataclass(frozen=True)
class GeneratedTask:
    public: PublicTask
    private: PrivateTask


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _task_seed(root_seed: int, partition: str, family: str, ordinal: int) -> int:
    material = f"{root_seed}:{partition}:{family}:{ordinal}".encode("ascii")
    return int.from_bytes(sha256(material).digest(), "big")


def _random_int_list(rng: random.Random, maximum_length: int = 18) -> list[int]:
    return [rng.randint(-80, 80) for _ in range(rng.randint(0, maximum_length))]


def _random_text(rng: random.Random, maximum_length: int = 36) -> str:
    alphabet = "aaabBcxyzXYZ019 _-"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, maximum_length)))


def _random_tokens(rng: random.Random, maximum_length: int = 16) -> str:
    vocabulary = ("aa", "b", "ccc", "x", "zero")
    tokens = [rng.choice(vocabulary) for _ in range(rng.randint(0, maximum_length))]
    separator = rng.choice((" ", "  ", "\t"))
    return separator.join(tokens)


def _property_inputs(
    rng: random.Random,
    edges: Sequence[tuple[JsonValue, ...]],
    factory: Callable[[random.Random], tuple[JsonValue, ...]],
    count: int,
) -> list[tuple[tuple[JsonValue, ...], str]]:
    values = [(args, "edge-distribution") for args in edges]
    while len(values) < count:
        values.append((factory(rng), "random-small"))
    return values[:count]


def _make_task(
    *,
    entry_point_nonce: bytes,
    partition: str,
    family: str,
    ordinal: int,
    parameter_names: tuple[str, ...],
    description: str,
    oracle: Oracle,
    regression_inputs: Sequence[tuple[JsonValue, ...]],
    property_inputs: Sequence[tuple[tuple[JsonValue, ...], str]],
) -> GeneratedTask:
    tier = int(family[1])
    task_id = f"simple-swe-v1/{partition}/tier-{tier}/{family}/{ordinal:03d}"
    entry_point_digest = sha256(
        b"simple-swe-entry-point-v1\0" + entry_point_nonce
    ).hexdigest()
    entry_point = f"solve_{entry_point_digest[:16]}"
    signature = ", ".join(parameter_names)
    prompt = (
        "Return exactly one complete Python function. Use only Python built-ins; "
        "do not emit Markdown, prose, tests, imports, or input/output examples.\n"
        "Implement this exact task:\n"
        f"def {entry_point}({signature}):\n"
        f"    \"\"\"{description}\"\"\""
    )
    regressions = tuple(
        HiddenCase(args=tuple(args), expected=oracle(*args), distribution="fixed-regression")
        for args in regression_inputs
    )
    properties = tuple(
        HiddenCase(args=tuple(args), expected=oracle(*args), distribution=distribution)
        for args, distribution in property_inputs
    )
    private_record = {
        "task_id": task_id,
        "entry_point": entry_point,
        "parameter_names": list(parameter_names),
        "regression_cases": [asdict(case) for case in regressions],
        "property_cases": [asdict(case) for case in properties],
    }
    test_sha256 = canonical_sha256(private_record)
    public_record = {
        "task_id": task_id,
        "partition": partition,
        "tier": tier,
        "family": family,
        "ordinal": ordinal,
        "entry_point": entry_point,
        "parameter_names": list(parameter_names),
        "prompt": prompt,
        "hidden_test_sha256": test_sha256,
    }
    task_sha256 = canonical_sha256(public_record)
    public = PublicTask(**public_record, task_sha256=task_sha256)
    private = PrivateTask(
        task_id=task_id,
        public_task_sha256=task_sha256,
        regression_cases=regressions,
        property_cases=properties,
        test_sha256=test_sha256,
    )
    return GeneratedTask(public=public, private=private)


TaskMaker = Callable[..., GeneratedTask]
FamilyBuilder = Callable[
    [random.Random, int, TaskMaker, str, str, int], GeneratedTask
]


def _build_scalar_clamp(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    limit = rng.randint(2, 12)
    oracle = lambda n: max(-limit, min(limit, n))
    regressions = ((-limit - 1,), (-limit,), (0,), (limit + 1,))
    properties = _property_inputs(
        rng,
        ((-100,), (-limit + 1,), (limit - 1,), (100,)),
        lambda r: (r.randint(-100, 100),),
        property_count,
    )
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("n",),
        description=f"Return n limited to the inclusive range {-limit} through {limit}.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_scalar_offset(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    delta = rng.choice(tuple(range(-9, 0)) + tuple(range(1, 10)))
    oracle = lambda n: n + delta
    regressions = ((0,), (1,), (-1,), (50,))
    properties = _property_inputs(
        rng,
        ((-100,), (0,), (100,)),
        lambda r: (r.randint(-100, 100),),
        property_count,
    )
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("n",),
        description=f"Return n plus the fixed offset {delta}.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_scalar_nonnegative(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    oracle = lambda n: max(0, n)
    regressions = ((-10,), (-1,), (0,), (9,))
    properties = _property_inputs(
        rng,
        ((-100,), (0,), (100,)),
        lambda r: (r.randint(-100, 100),),
        property_count,
    )
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("n",),
        description="Return n when it is nonnegative, otherwise return zero.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_list_map_add(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    delta = rng.choice((-5, -3, -1, 1, 3, 5))
    oracle = lambda values: [value + delta for value in values]
    regressions = (([],), ([0],), ([-1, 1],), ([4, 4, -2],))
    properties = _property_inputs(
        rng,
        (([],), ([0] * 18,), ([-80, 80],)),
        lambda r: (_random_int_list(r),),
        property_count,
    )
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("values",),
        description=f"Return a new list with {delta} added to every integer in values.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_list_map_abs(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    oracle = lambda values: [abs(value) for value in values]
    regressions = (([],), ([0],), ([-1, 1],), ([-9, -2, 5],))
    properties = _property_inputs(
        rng,
        (([],), ([-80, 0, 80],), ([0] * 18,)),
        lambda r: (_random_int_list(r),),
        property_count,
    )
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("values",),
        description="Return a new list containing the absolute value of each integer in values.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_string_filter(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    label, predicate = rng.choice(
        (
            ("ASCII vowels", lambda char: char in "aeiouAEIOU"),
            ("ASCII decimal digits", lambda char: char in "0123456789"),
            ("lowercase ASCII letters", lambda char: "a" <= char <= "z"),
        )
    )
    reverse = family.startswith("t2_")
    oracle = lambda text: "".join(char for char in text if predicate(char))[
        :: -1 if reverse else 1
    ]
    regressions = (("",), ("abcXYZ019",), ("___",), ("a A 9",))
    properties = _property_inputs(
        rng,
        (("",), (" " * 36,), ("aA0_" * 9,)),
        lambda r: (_random_text(r),),
        property_count,
    )
    action = "then reverse the retained characters" if reverse else "in their original order"
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("text",),
        description=f"Keep only {label} from text {action}; return the resulting string.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_clamp_then_sum(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    limit = rng.randint(2, 10)
    oracle = lambda values: sum(max(-limit, min(limit, value)) for value in values)
    regressions = (([],), ([0],), ([-limit - 1, limit + 1],), ([1, 2, 3],))
    properties = _property_inputs(
        rng,
        (([],), ([-80, 80],), ([0] * 18,)),
        lambda r: (_random_int_list(r),),
        property_count,
    )
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("values",),
        description=f"Limit every integer to {-limit} through {limit}, then return their sum.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_map_then_count(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    delta = rng.choice((-4, -2, 2, 4))
    threshold = rng.randint(-3, 3)
    oracle = lambda values: sum(value + delta > threshold for value in values)
    regressions = (([],), ([threshold - delta],), ([threshold - delta + 1],), ([-9, 0, 9],))
    properties = _property_inputs(
        rng,
        (([],), ([threshold - delta] * 18,), ([-80, 80],)),
        lambda r: (_random_int_list(r),),
        property_count,
    )
    return make_task(
        partition=partition,
        family=family,
        ordinal=ordinal,
        parameter_names=("values",),
        description=f"Add {delta} to each integer and count results strictly greater than {threshold}.",
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )


def _build_digit_sum(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    oracle = lambda n: sum(int(char) for char in str(abs(n)))
    regressions = ((0,), (7,), (-42,), (909,))
    properties = _property_inputs(
        rng,
        ((0,), (-999999,), (999999,)),
        lambda r: (r.randint(-999999, 999999),),
        property_count,
    )
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("n",),
        description="Return the sum of the decimal digits of the absolute value of n.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_stable_parity_select(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    parity = rng.randint(0, 1)
    oracle = lambda values: [value for value in values if value % 2 == parity]
    regressions = (([],), ([0],), ([1],), ([-3, -2, -1, 0, 1, 2, 3],))
    properties = _property_inputs(rng, (([],), ([-80, 80],), ([parity] * 18,)), lambda r: (_random_int_list(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("values",),
        description=f"Return the integers whose remainder modulo 2 is {parity}, preserving order.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_longest_character_run(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    def oracle(text: str) -> int:
        best = current = 0
        previous = None
        for char in text:
            current = current + 1 if char == previous else 1
            best = max(best, current)
            previous = char
        return best
    regressions = (("",), ("a",), ("ab",), ("aaabbcccc",))
    properties = _property_inputs(rng, (("",), ("a" * 36,), ("abab" * 9,)), lambda r: (_random_text(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("text",),
        description="Return the length of the longest consecutive run of one identical character; return zero for empty text.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_nonzero_digit_product(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    def oracle(n: int) -> int:
        digits = [int(char) for char in str(abs(n)) if char != "0"]
        result = 1
        for digit in digits:
            result *= digit
        return result if digits else 0
    regressions = ((0,), (7,), (-204,), (909,))
    properties = _property_inputs(rng, ((0,), (-100000,), (999999,)), lambda r: (r.randint(-999999, 999999),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("n",),
        description="Multiply the nonzero decimal digits of abs(n); return zero when there are no nonzero digits.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_stable_range_select(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    lower = rng.randint(-10, -1)
    upper = rng.randint(1, 10)
    oracle = lambda values: [value for value in values if lower <= value <= upper]
    regressions = (([],), ([lower - 1, lower],), ([upper, upper + 1],), ([-20, 0, 20],))
    properties = _property_inputs(rng, (([],), ([lower, upper] * 9,), ([-80, 80],)), lambda r: (_random_int_list(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("values",),
        description=f"Keep integers in the inclusive range {lower} through {upper}, preserving order.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_longest_target_run(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    target = rng.choice("abc01")
    def oracle(text: str) -> int:
        best = current = 0
        for char in text:
            current = current + 1 if char == target else 0
            best = max(best, current)
        return best
    regressions = (("",), (target,), (target * 4,), (f"x{target}{target}x",))
    properties = _property_inputs(rng, (("",), (target * 36,), ((target + "x") * 18,)), lambda r: (_random_text(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("text",),
        description=f"Return the longest consecutive run of the character {target!r}; return zero if absent.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_digit_reduce_mod(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    modulus = rng.randint(2, 9)
    oracle = lambda n: sum(int(char) for char in str(abs(n))) % modulus
    regressions = ((0,), (modulus,), (-42,), (999,))
    properties = _property_inputs(rng, ((0,), (-999999,), (999999,)), lambda r: (r.randint(-999999, 999999),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("n",),
        description=f"Sum the decimal digits of abs(n), then return that sum modulo {modulus}.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_select_then_span(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    parity = rng.randint(0, 1)
    def oracle(values: list[int]) -> int:
        selected = [value for value in values if value % 2 == parity]
        return max(selected) - min(selected) if selected else 0
    regressions = (([],), ([parity],), ([0, 1],), ([-9, -2, 4, 7],))
    properties = _property_inputs(rng, (([],), ([parity] * 18,), ([-80, 80],)), lambda r: (_random_int_list(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("values",),
        description=f"Select integers with remainder {parity} modulo 2 and return max minus min; return zero if none.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_run_length_signature(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    def oracle(text: str) -> list[int]:
        lengths: list[int] = []
        for char in text:
            if lengths and char == text[sum(lengths) - 1]:
                lengths[-1] += 1
            else:
                lengths.append(1)
        return lengths
    regressions = (("",), ("a",), ("ab",), ("aaabbc",))
    properties = _property_inputs(rng, (("",), ("a" * 36,), ("ab" * 18,)), lambda r: (_random_text(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("text",),
        description="Return a list with the lengths of consecutive identical-character runs, in order.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_bounded_scalar_classifier(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    limit = rng.randint(2, 12)
    oracle = lambda n: -1 if n < -limit else (1 if n > limit else 0)
    regressions = ((-limit - 1,), (-limit,), (limit,), (limit + 1,))
    properties = _property_inputs(rng, ((-100,), (0,), (100,)), lambda r: (r.randint(-100, 100),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("n",),
        description=f"Return -1 below {-limit}, 1 above {limit}, and 0 inside that inclusive band.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_adjacent_equal_count(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    oracle = lambda values: sum(left == right for left, right in zip(values, values[1:]))
    regressions = (([],), ([1],), ([1, 1],), ([1, 1, 2, 2, 2],))
    properties = _property_inputs(rng, (([],), ([0] * 18,), (list(range(18)),)), lambda r: (_random_int_list(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("values",),
        description="Count adjacent pairs of equal integers; overlapping pairs each count once.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_token_boundary_length(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    def oracle(text: str) -> int:
        words = text.split()
        return len(words[0]) + len(words[-1]) if words else 0
    regressions = (("",), ("a",), ("a bb",), ("  aa   bbb  ",))
    properties = _property_inputs(rng, (("",), (" " * 36,), ("aa " * 18,)), lambda r: (_random_tokens(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("text",),
        description="Split on whitespace and return first-word length plus last-word length; return zero when no words exist.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_bounded_divisibility_label(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    bound = rng.randint(20, 60)
    divisor = rng.randint(2, 9)
    oracle = lambda n: "out" if abs(n) > bound else ("divisible" if n % divisor == 0 else "other")
    regressions = ((-bound - 1,), (0,), (divisor,), (bound + 1,))
    properties = _property_inputs(rng, ((-100,), (0,), (100,)), lambda r: (r.randint(-100, 100),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("n",),
        description=f"Return 'out' when abs(n) exceeds {bound}; otherwise return 'divisible' when n is divisible by {divisor}, else 'other'.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_adjacent_differences(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    oracle = lambda values: [right - left for left, right in zip(values, values[1:])]
    regressions = (([],), ([1],), ([1, 3],), ([4, 1, 1, -2],))
    properties = _property_inputs(rng, (([],), ([0] * 18,), ([-80, 80],)), lambda r: (_random_int_list(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("values",),
        description="Return a list of next value minus current value for every adjacent pair.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_word_length_filter(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    lower = rng.randint(1, 3)
    upper = rng.randint(lower + 1, 7)
    oracle = lambda text: [word for word in text.split() if lower <= len(word) <= upper]
    regressions = (("",), ("a bb ccc",), ("   aa   bbb ",), ("xxxxxxxx short",))
    properties = _property_inputs(rng, (("",), (" " * 36,), ("aa " * 18,)), lambda r: (_random_tokens(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("text",),
        description=f"Split on whitespace and return words whose lengths are from {lower} through {upper}, preserving order.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_clamped_window_sums(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    limit = rng.randint(3, 10)
    width = rng.randint(2, 5)
    def oracle(values: list[int]) -> list[int]:
        clamped = [max(-limit, min(limit, value)) for value in values]
        return [sum(clamped[index : index + width]) for index in range(len(clamped) - width + 1)]
    regressions = (([],), ([1] * (width - 1),), ([1] * width,), ([-80, 0, 80, 1, 2, 3],))
    properties = _property_inputs(rng, (([],), ([0] * 18,), ([-80, 80],)), lambda r: (_random_int_list(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("values",),
        description=f"Limit values to {-limit} through {limit}, then return sums of every consecutive window of width {width}; return an empty list if too short.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_filtered_adjacent_changes(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    bound = rng.randint(4, 15)
    def oracle(values: list[int]) -> int:
        kept = [value for value in values if abs(value) <= bound]
        return sum(left != right for left, right in zip(kept, kept[1:]))
    regressions = (([],), ([bound + 1],), ([1, 1, 2],), ([-80, -1, 1, 80],))
    properties = _property_inputs(rng, (([],), ([0] * 18,), ([-80, 80],)), lambda r: (_random_int_list(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("values",),
        description=f"Discard integers with abs(value) greater than {bound}, then count changes between adjacent retained values.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


def _build_token_run_signature(
    rng: random.Random,
    property_count: int,
    make_task: TaskMaker,
    partition: str,
    family: str,
    ordinal: int,
) -> GeneratedTask:
    def oracle(text: str) -> list[int]:
        words = text.split()
        lengths: list[int] = []
        previous = None
        for word in words:
            if lengths and word == previous:
                lengths[-1] += 1
            else:
                lengths.append(1)
            previous = word
        return lengths
    regressions = (("",), ("aa",), ("aa aa b",), ("x x x b b x",))
    properties = _property_inputs(rng, (("",), ("aa " * 18,), ("aa b " * 9,)), lambda r: (_random_tokens(r),), property_count)
    return make_task(
        partition=partition, family=family, ordinal=ordinal, parameter_names=("text",),
        description="Split on whitespace and return the lengths of consecutive runs of identical tokens.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


_FAMILY_BUILDERS: dict[str, FamilyBuilder] = {
    "t0_scalar_clamp": _build_scalar_clamp,
    "t0_scalar_offset": _build_scalar_offset,
    "t0_scalar_nonnegative": _build_scalar_nonnegative,
    "t1_list_map_add": _build_list_map_add,
    "t1_list_map_abs": _build_list_map_abs,
    "t1_string_filter_class": _build_string_filter,
    "t2_filter_then_reverse": _build_string_filter,
    "t2_clamp_then_sum": _build_clamp_then_sum,
    "t2_map_then_count": _build_map_then_count,
    "t0_digit_sum": _build_digit_sum,
    "t0_stable_parity_select": _build_stable_parity_select,
    "t0_longest_character_run": _build_longest_character_run,
    "t1_nonzero_digit_product": _build_nonzero_digit_product,
    "t1_stable_range_select": _build_stable_range_select,
    "t1_longest_target_run": _build_longest_target_run,
    "t2_digit_reduce_mod": _build_digit_reduce_mod,
    "t2_select_then_span": _build_select_then_span,
    "t2_run_length_signature": _build_run_length_signature,
    "t0_bounded_scalar_classifier": _build_bounded_scalar_classifier,
    "t0_adjacent_equal_count": _build_adjacent_equal_count,
    "t0_token_boundary_length": _build_token_boundary_length,
    "t1_bounded_divisibility_label": _build_bounded_divisibility_label,
    "t1_adjacent_differences": _build_adjacent_differences,
    "t1_word_length_filter": _build_word_length_filter,
    "t2_clamped_window_sums": _build_clamped_window_sums,
    "t2_filtered_adjacent_changes": _build_filtered_adjacent_changes,
    "t2_token_run_signature": _build_token_run_signature,
}


def _build_task(
    protocol: ProtocolSpec,
    partition: str,
    family: str,
    ordinal: int,
    root_seed: int,
) -> GeneratedTask:
    rng = random.Random(_task_seed(root_seed, partition, family, ordinal))
    property_count = protocol.property_cases_per_task
    entry_point_nonce = sha256(
        f"{root_seed}:{partition}:{family}:{ordinal}:entry-point-v1".encode("ascii")
    ).digest()
    make_task = partial(_make_task, entry_point_nonce=entry_point_nonce)
    builder = _FAMILY_BUILDERS.get(family)
    if builder is None:
        raise ValueError(f"protocol references unknown task family {family}")
    return builder(rng, property_count, make_task, partition, family, ordinal)


def _generate_partition(
    protocol: ProtocolSpec, partition: str, root_seed: int
) -> tuple[GeneratedTask, ...]:
    specification = protocol.partitions[partition]
    tasks = tuple(
        _build_task(protocol, partition, family, ordinal, root_seed)
        for family, quota in specification.family_quotas.items()
        for ordinal in range(quota)
    )
    if len(tasks) != specification.task_count:
        raise RuntimeError("generated task count differs from frozen protocol")
    if any(
        len(task.private.regression_cases) != protocol.regression_cases_per_task
        or len(task.private.property_cases) != protocol.property_cases_per_task
        for task in tasks
    ):
        raise RuntimeError("generated hidden-test counts differ from frozen protocol")
    return tasks


def generate_visible_partition(
    protocol: ProtocolSpec, partition: str
) -> tuple[GeneratedTask, ...]:
    """Generate only pre-freeze train/dev material; eval is reveal-only."""
    if partition == "eval":
        raise PermissionError("eval tasks require the post-freeze reveal path")
    if partition not in {"train", "dev"}:
        raise ValueError("visible partition must be train or dev")
    seed = protocol.partitions[partition].seed
    if seed is None:
        raise RuntimeError("visible partition is missing its frozen seed")
    return _generate_partition(protocol, partition, seed)
