from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
from types import MappingProxyType
from typing import Callable, Sequence, TypeAlias

from humanbrain.simple_swe.protocol import ProtocolSpec
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
Oracle = Callable[..., JsonValue]


@dataclass(frozen=True)
class HiddenCase:
    args: tuple[JsonValue, ...]
    expected: JsonValue
    distribution: str


@dataclass(frozen=True, slots=True)
class CallableType:
    """One name-independent callable type shared by a Simple-SWE family."""

    parameters: tuple[ProgramParameter, ...]
    result: ProgramType

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, tuple) or not all(
            isinstance(parameter, ProgramParameter) for parameter in self.parameters
        ):
            raise TypeError("callable parameters must be ProgramParameter values")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("callable parameter names must be unique")
        if not isinstance(self.result, ProgramType):
            raise TypeError("callable result must be a ProgramType")

    def bind(self, entry_point: str) -> TypedSignature:
        return TypedSignature(
            entry_point=entry_point,
            parameters=self.parameters,
            result=self.result,
        )


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    partition: str
    tier: int
    family: str
    ordinal: int
    signature: TypedSignature
    prompt: str
    hidden_test_sha256: str
    task_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.signature, TypedSignature):
            raise TypeError("public task signature must be typed")


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


def make_task(
    *,
    partition: str,
    family: str,
    ordinal: int,
    signature: TypedSignature,
    description: str,
    oracle: Oracle,
    regression_inputs: Sequence[tuple[JsonValue, ...]],
    property_inputs: Sequence[tuple[tuple[JsonValue, ...], str]],
) -> GeneratedTask:
    """Construct one committed task bundle for visible development tooling."""
    tier = int(family[1])
    task_id = f"simple-swe-v1/{partition}/tier-{tier}/{family}/{ordinal:03d}"
    parameter_names = tuple(parameter.name for parameter in signature.parameters)
    python_parameters = ", ".join(parameter_names)
    prompt = (
        "Return exactly one complete Python function. Use only Python built-ins; "
        "do not emit Markdown, prose, tests, imports, or input/output examples.\n"
        "Implement this exact task:\n"
        f"def {signature.entry_point}({python_parameters}):\n"
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
        "signature": signature.to_record(),
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
        "signature": signature.to_record(),
        "prompt": prompt,
        "hidden_test_sha256": test_sha256,
    }
    task_sha256 = canonical_sha256(public_record)
    public = PublicTask(
        task_id=task_id,
        partition=partition,
        tier=tier,
        family=family,
        ordinal=ordinal,
        signature=signature,
        prompt=prompt,
        hidden_test_sha256=test_sha256,
        task_sha256=task_sha256,
    )
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
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
        partition=partition, family=family, ordinal=ordinal,
        description="Split on whitespace and return the lengths of consecutive runs of identical tokens.",
        oracle=oracle, regression_inputs=regressions, property_inputs=properties,
    )


@dataclass(frozen=True, slots=True)
class _FamilySpec:
    builder: FamilyBuilder
    callable_type: CallableType


def _unary_callable(
    parameter_name: str, parameter_type: ProgramType, result: ProgramType
) -> CallableType:
    return CallableType(
        parameters=(ProgramParameter(parameter_name, parameter_type),),
        result=result,
    )


_INT = ProgramType.Int()
_STRING = ProgramType.String()
_INT_LIST = ProgramType.List(_INT)
_STRING_LIST = ProgramType.List(_STRING)

_FAMILY_SPECS: Mapping[str, _FamilySpec] = MappingProxyType({
    "t0_scalar_clamp": _FamilySpec(
        _build_scalar_clamp, _unary_callable("n", _INT, _INT)
    ),
    "t0_scalar_offset": _FamilySpec(
        _build_scalar_offset, _unary_callable("n", _INT, _INT)
    ),
    "t0_scalar_nonnegative": _FamilySpec(
        _build_scalar_nonnegative, _unary_callable("n", _INT, _INT)
    ),
    "t1_list_map_add": _FamilySpec(
        _build_list_map_add, _unary_callable("values", _INT_LIST, _INT_LIST)
    ),
    "t1_list_map_abs": _FamilySpec(
        _build_list_map_abs, _unary_callable("values", _INT_LIST, _INT_LIST)
    ),
    "t1_string_filter_class": _FamilySpec(
        _build_string_filter, _unary_callable("text", _STRING, _STRING)
    ),
    "t2_filter_then_reverse": _FamilySpec(
        _build_string_filter, _unary_callable("text", _STRING, _STRING)
    ),
    "t2_clamp_then_sum": _FamilySpec(
        _build_clamp_then_sum, _unary_callable("values", _INT_LIST, _INT)
    ),
    "t2_map_then_count": _FamilySpec(
        _build_map_then_count, _unary_callable("values", _INT_LIST, _INT)
    ),
    "t0_digit_sum": _FamilySpec(
        _build_digit_sum, _unary_callable("n", _INT, _INT)
    ),
    "t0_stable_parity_select": _FamilySpec(
        _build_stable_parity_select,
        _unary_callable("values", _INT_LIST, _INT_LIST),
    ),
    "t0_longest_character_run": _FamilySpec(
        _build_longest_character_run, _unary_callable("text", _STRING, _INT)
    ),
    "t1_nonzero_digit_product": _FamilySpec(
        _build_nonzero_digit_product, _unary_callable("n", _INT, _INT)
    ),
    "t1_stable_range_select": _FamilySpec(
        _build_stable_range_select,
        _unary_callable("values", _INT_LIST, _INT_LIST),
    ),
    "t1_longest_target_run": _FamilySpec(
        _build_longest_target_run, _unary_callable("text", _STRING, _INT)
    ),
    "t2_digit_reduce_mod": _FamilySpec(
        _build_digit_reduce_mod, _unary_callable("n", _INT, _INT)
    ),
    "t2_select_then_span": _FamilySpec(
        _build_select_then_span, _unary_callable("values", _INT_LIST, _INT)
    ),
    "t2_run_length_signature": _FamilySpec(
        _build_run_length_signature,
        _unary_callable("text", _STRING, _INT_LIST),
    ),
    "t0_bounded_scalar_classifier": _FamilySpec(
        _build_bounded_scalar_classifier, _unary_callable("n", _INT, _INT)
    ),
    "t0_adjacent_equal_count": _FamilySpec(
        _build_adjacent_equal_count, _unary_callable("values", _INT_LIST, _INT)
    ),
    "t0_token_boundary_length": _FamilySpec(
        _build_token_boundary_length, _unary_callable("text", _STRING, _INT)
    ),
    "t1_bounded_divisibility_label": _FamilySpec(
        _build_bounded_divisibility_label, _unary_callable("n", _INT, _STRING)
    ),
    "t1_adjacent_differences": _FamilySpec(
        _build_adjacent_differences,
        _unary_callable("values", _INT_LIST, _INT_LIST),
    ),
    "t1_word_length_filter": _FamilySpec(
        _build_word_length_filter,
        _unary_callable("text", _STRING, _STRING_LIST),
    ),
    "t2_clamped_window_sums": _FamilySpec(
        _build_clamped_window_sums,
        _unary_callable("values", _INT_LIST, _INT_LIST),
    ),
    "t2_filtered_adjacent_changes": _FamilySpec(
        _build_filtered_adjacent_changes,
        _unary_callable("values", _INT_LIST, _INT),
    ),
    "t2_token_run_signature": _FamilySpec(
        _build_token_run_signature,
        _unary_callable("text", _STRING, _INT_LIST),
    ),
})


def family_callable_types() -> Mapping[str, CallableType]:
    """Return the immutable public callable type for every frozen task family."""
    return MappingProxyType(
        {
            family: specification.callable_type
            for family, specification in _FAMILY_SPECS.items()
        }
    )


def _build_task(
    protocol: ProtocolSpec,
    partition: str,
    family: str,
    ordinal: int,
    root_seed: int,
) -> GeneratedTask:
    rng = random.Random(_task_seed(root_seed, partition, family, ordinal))
    property_count = protocol.property_cases_per_task
    specification = _FAMILY_SPECS.get(family)
    if specification is None:
        raise ValueError(f"protocol references unknown task family {family}")
    entry_point_nonce = sha256(
        f"{root_seed}:{partition}:{family}:{ordinal}:entry-point-v1".encode("ascii")
    ).digest()
    entry_point_digest = sha256(
        b"simple-swe-entry-point-v1\0" + entry_point_nonce
    ).hexdigest()
    signature = specification.callable_type.bind(
        f"solve_{entry_point_digest[:16]}"
    )
    task_factory = partial(make_task, signature=signature)
    return specification.builder(
        rng, property_count, task_factory, partition, family, ordinal
    )


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
