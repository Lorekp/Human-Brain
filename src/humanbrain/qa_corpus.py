"""Deterministic verse-context QA corpus for the QA-v1 gate.

The typed-hole seam cannot host free-text answers (string holes have zero
inhabiting primitives), so QA-v1 extends the LANGUAGE side instead: this
module materializes question/answer tasks whose ``qa_prompt`` framing feeds
``RecurrentCellLanguageModel.grow`` / ``complete`` unchanged.

v1 ships a small bundled FIXTURE corpus of thirty KJV verse-context QA
triples. This is clearly a fixture: it exists so tests and gates have a
byte-stable, answer-derivable corpus until the freeze-time PG10 pipeline
described in ``docs/research/cell-lm-corpus-bible.md`` lands — that doc
(added mid-flight to this branch) owns the post-freeze corpus source,
split, and its own ``\nQ: ``/``\nA: `` manifest delimiters; ``qa_prompt``
here is the single framing swap point when those manifests arrive. Every
fixture answer occurs verbatim (case-folded) inside its context verses.

Materialization discipline mirrors ``benchmarks/ladder_experiments``
``build_curriculum`` over ``typed_composition._materialize_tasks``:
seed-keyed identifier/order domains, sha256 presentation-order sort keys,
dense ordinals with per-cohort offsets, opaque derived task ids, and one
canonical-JSON curriculum digest recorded in every run record.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

from humanbrain.typed_composition import derive_opaque_identifier

QA_SCHEMA_VERSION = "humanbrain/qa-corpus/v1"
CORPUS_SOURCE = "fixture:v1-kjv-verse-contexts"
DEFAULT_CALIBRATION_TASK_COUNT = 6

FIXTURE_NOTICE = (
    "QA-v1 fixture corpus: thirty hand-written KJV verse-context QA triples "
    "(public-domain text). Not the full Bible corpus; answers are verbatim "
    "substrings of their contexts by construction."
)


@dataclass(frozen=True)
class QATask:
    """One held evidence record; only ``qa_prompt`` text reaches cells."""

    cohort: str
    qa_id: str
    ordinal: int
    references: tuple[str, ...]
    context: tuple[str, ...]
    question: str
    answer: str


# --- begin fixture data (clearly marked; see FIXTURE_NOTICE) ----------------

FIXTURE_CORPUS: tuple[dict[str, object], ...] = (
    {"references": ("Genesis 1:1",), "verses": ("Genesis 1:1 In the beginning God created the heaven and the earth.",), "question": "What did God create in the beginning?", "answer": "the heaven and the earth"},
    {"references": ("John 1:1",), "verses": ("John 1:1 In the beginning was the Word, and the Word was with God, and the Word was God.",), "question": "What was in the beginning according to John?", "answer": "the Word"},
    {"references": ("Psalm 23:1", "Psalm 23:2"), "verses": ("Psalm 23:1 The LORD is my shepherd; I shall not want.", "Psalm 23:2 He maketh me to lie down in green pastures: he leadeth me beside the still waters."), "question": "Who is my shepherd?", "answer": "the LORD"},
    {"references": ("John 3:16",), "verses": ("John 3:16 For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.",), "question": "Whom did God give for the world?", "answer": "his only begotten Son"},
    {"references": ("Proverbs 3:5",), "verses": ("Proverbs 3:5 Trust in the LORD with all thine heart; and lean not unto thine own understanding.",), "question": "With what should I trust in the LORD?", "answer": "all thine heart"},
    {"references": ("Matthew 5:3",), "verses": ("Matthew 5:3 Blessed are the poor in spirit: for theirs is the kingdom of heaven.",), "question": "Whose is the kingdom of heaven?", "answer": "the poor in spirit"},
    {"references": ("Romans 6:23",), "verses": ("Romans 6:23 For the wages of sin is death; but the gift of God is eternal life through Jesus Christ our Lord.",), "question": "What are the wages of sin?", "answer": "death"},
    {"references": ("Genesis 1:3",), "verses": ("Genesis 1:3 And God said, Let there be light: and there was light.",), "question": "What did God command into being?", "answer": "light"},
    {"references": ("John 11:35",), "verses": ("John 11:35 Jesus wept.",), "question": "Who wept?", "answer": "Jesus"},
    {"references": ("Philippians 4:13",), "verses": ("Philippians 4:13 I can do all things through Christ which strengtheneth me.",), "question": "Through whom can I do all things?", "answer": "Christ"},
    {"references": ("Joshua 1:9",), "verses": ("Joshua 1:9 Be strong and of a good courage; for the LORD thy God is with thee whithersoever thou goest.",), "question": "Who is with thee whithersoever thou goest?", "answer": "the LORD thy God"},
    {"references": ("Isaiah 40:31",), "verses": ("Isaiah 40:31 But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles; they shall run, and not be weary; and they shall walk, and not faint.",), "question": "Who shall renew their strength?", "answer": "they that wait upon the LORD"},
    {"references": ("Psalm 119:105",), "verses": ("Psalm 119:105 Thy word is a lamp unto my feet, and a light unto my path.",), "question": "What is a lamp unto my feet?", "answer": "Thy word"},
    {"references": ("Hebrews 11:1",), "verses": ("Hebrews 11:1 Now faith is the substance of things hoped for, the evidence of things not seen.",), "question": "According to Hebrews, what is faith?", "answer": "the substance of things hoped for"},
    {"references": ("Ephesians 4:32",), "verses": ("Ephesians 4:32 And be ye kind one to another, tenderhearted, forgiving one another, even as God for Christ's sake hath forgiven you.",), "question": "How should ye be one to another?", "answer": "kind"},
    {"references": ("Galatians 5:22",), "verses": ("Galatians 5:22 But the fruit of the Spirit is love, joy, peace, longsuffering, gentleness, goodness, faith,",), "question": "Name three things that are the fruit of the Spirit.", "answer": "love, joy, peace"},
    {"references": ("Proverbs 22:6",), "verses": ("Proverbs 22:6 Train up a child in the way he should go: and when he is old, he will not depart from it.",), "question": "Whom should I train up?", "answer": "a child"},
    {"references": ("John 14:6",), "verses": ("John 14:6 Jesus saith unto him, I am the way, the truth, and the life: no man cometh unto the Father, but by me.",), "question": "What did Jesus say he is?", "answer": "the way, the truth, and the life"},
    {"references": ("Numbers 6:24", "Numbers 6:25", "Numbers 6:26"), "verses": ("Numbers 6:24 The LORD bless thee, and keep thee:", "Numbers 6:25 The LORD make his face shine upon thee, and be gracious unto thee:", "Numbers 6:26 The LORD lift up his countenance upon thee, and give thee peace."), "question": "Who shall bless thee and keep thee?", "answer": "The LORD"},
    {"references": ("Matthew 6:33",), "verses": ("Matthew 6:33 But seek ye first the kingdom of God, and his righteousness; and all these things shall be added unto you.",), "question": "What should ye seek first?", "answer": "the kingdom of God"},
    {"references": ("Romans 8:28",), "verses": ("Romans 8:28 And we know that all things work together for good to them that love God, to them who are the called according to his purpose.",), "question": "For whom do all things work together for good?", "answer": "them that love God"},
    {"references": ("Ecclesiastes 3:1",), "verses": ("Ecclesiastes 3:1 To every thing there is a season, and a time to every purpose under the heaven:",), "question": "What is there to every thing?", "answer": "a season"},
    {"references": ("Luke 6:31",), "verses": ("Luke 6:31 And as ye would that men should do to you, do ye also to them likewise.",), "question": "State the rule Luke gives for treating others.", "answer": "as ye would that men should do to you, do ye also to them likewise"},
    {"references": ("Jeremiah 29:11",), "verses": ("Jeremiah 29:11 For I know the thoughts that I think toward you, saith the LORD, thoughts of peace, and not of evil, to give you an expected end.",), "question": "Who speaks thoughts of peace toward you here?", "answer": "the LORD"},
    {"references": ("Psalm 46:10",), "verses": ("Psalm 46:10 Be still, and know that I am God: I will be exalted among the heathen, I will be exalted in the earth.",), "question": "What does the psalm say to know?", "answer": "that I am God"},
    {"references": ("Matthew 22:37", "Matthew 22:38", "Matthew 22:39"), "verses": ("Matthew 22:37 Jesus said unto him, Thou shalt love the Lord thy God with all thy heart, and with all thy soul, and with all thy mind.", "Matthew 22:38 This is the first and great commandment.", "Matthew 22:39 And the second is like unto it, Thou shalt love thy neighbour as thyself."), "question": "With what shalt thou love the Lord thy God?", "answer": "all thy heart"},
    {"references": ("Exodus 20:13", "Exodus 20:14", "Exodus 20:15"), "verses": ("Exodus 20:13 Thou shalt not kill.", "Exodus 20:14 Thou shalt not commit adultery.", "Exodus 20:15 Thou shalt not steal."), "question": "According to Exodus 20:15, what shalt thou not do?", "answer": "steal"},
    {"references": ("1 Corinthians 13:4",), "verses": ("1 Corinthians 13:4 Charity suffereth long, and is kind; charity envieth not; charity vaunteth not itself, is not puffed up,",), "question": "What suffereth long and is kind?", "answer": "Charity"},
    {"references": ("Genesis 3:19",), "verses": ("Genesis 3:19 In the sweat of thy face shalt thou eat bread, till thou return unto the ground; for out of it wast thou taken: for dust thou art, and unto dust shalt thou return.",), "question": "Unto what shalt thou return?", "answer": "dust"},
    {"references": ("Psalm 121:1", "Psalm 121:2"), "verses": ("Psalm 121:1 I will lift up mine eyes unto the hills, from whence cometh my help.", "Psalm 121:2 My help cometh from the LORD, which made heaven and earth."), "question": "From whom cometh my help?", "answer": "the LORD"},
)

# --- end fixture data --------------------------------------------------------


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_entry(index: int, entry: dict[str, object]) -> None:
    label = f"fixture entry {index}"
    references = entry.get("references")
    verses = entry.get("verses")
    question = entry.get("question")
    answer = entry.get("answer")
    if (
        not isinstance(references, tuple)
        or not references
        or not all(isinstance(item, str) and item for item in references)
    ):
        raise ValueError(f"{label} needs a non-empty tuple of reference strings")
    if (
        not isinstance(verses, tuple)
        or not verses
        or not all(isinstance(item, str) and item for item in verses)
    ):
        raise ValueError(f"{label} needs a non-empty tuple of verse strings")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{label} needs a non-empty question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{label} needs a non-empty answer")
    context_text = " ".join(verses).casefold()
    if answer.casefold() not in context_text:
        raise ValueError(
            f"{label} answer {answer!r} is not derivable from its context verses"
        )


def load_corpus() -> tuple[dict[str, object], ...]:
    """Return the validated fixture corpus entries in fixed order."""
    seen: set[str] = set()
    for index, entry in enumerate(FIXTURE_CORPUS):
        _validate_entry(index, entry)
        for reference in entry["references"]:
            if reference in seen:
                raise ValueError(f"fixture entry {index} repeats reference {reference!r}")
            seen.add(reference)
    return FIXTURE_CORPUS


def corpus_record() -> dict[str, object]:
    """Canonical JSON-ready record of the whole fixture corpus."""
    return {
        "entries": [
            {
                "references": list(entry["references"]),
                "verses": list(entry["verses"]),
                "question": entry["question"],
                "answer": entry["answer"],
            }
            for entry in load_corpus()
        ]
    }


def corpus_digest() -> str:
    """SHA-256 over the canonical JSON of the validated fixture corpus."""
    return sha256(_canonical_json(corpus_record())).hexdigest()


def qa_prompt(context: Sequence[str], question: str) -> str:
    """Prompt-last framing: context verses, then the question, then ``answer:``."""
    lines = [str(verse) for verse in context]
    lines.append(f"Question: {question}")
    lines.append("answer:")
    return "\n".join(lines)


def prompt_for_task(task: QATask) -> str:
    return qa_prompt(task.context, task.question)


def growth_text(task: QATask) -> str:
    """Growth text pairs the prompt with its answer: ``<prompt> <answer>``."""
    return prompt_for_task(task) + " " + task.answer


def _task_sort_key(domain: str, task_id: str) -> bytes:
    """Mirror typed_composition._task_sort_key presentation-order keys."""
    return sha256(domain.encode("utf-8") + b"\x00" + task_id.encode("utf-8")).digest()


def _materialize(
    specs: list[tuple[str, dict[str, object]]],
    *,
    cohort: str,
    order_domain: str,
    ordinal_offset: int,
) -> list[QATask]:
    ordered = sorted(specs, key=lambda item: _task_sort_key(order_domain, item[0]))
    tasks: list[QATask] = []
    for index, (task_id, spec) in enumerate(ordered):
        ordinal = ordinal_offset + index
        tasks.append(
            QATask(
                cohort=cohort,
                qa_id=derive_opaque_identifier(order_domain + "/qa-id", ordinal),
                ordinal=ordinal,
                references=tuple(str(item) for item in spec["references"]),  # type: ignore[arg-type]
                context=tuple(str(item) for item in spec["verses"]),  # type: ignore[arg-type]
                question=str(spec["question"]),
                answer=str(spec["answer"]),
            )
        )
    return tasks


def build_curriculum(
    seed: int,
    *,
    calibration_task_count: int = DEFAULT_CALIBRATION_TASK_COUNT,
) -> dict[str, object]:
    """Materialize one fresh seed-keyed TRAIN/CALIBRATION cohort pair.

    Mirrors ladder_experiments.build_curriculum: two pools drawn from one
    global sha256 order, each ordered by its own order domain, train ordinals
    from zero and calibration ordinals offset past train, opaque identifiers
    derived per ordinal, and one canonical-JSON curriculum SHA-256.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not 1 <= calibration_task_count < len(FIXTURE_CORPUS):
        raise ValueError("calibration_task_count must leave both cohorts non-empty")
    entries = load_corpus()
    specs = [
        (f"qa-{index:03d}", dict(entry)) for index, entry in enumerate(entries)
    ]
    identifier_domain = f"{QA_SCHEMA_VERSION}/identifier/seed/{seed}"
    order_domain = f"{QA_SCHEMA_VERSION}/task-order/seed/{seed}"
    global_order = sorted(
        specs, key=lambda item: _task_sort_key(f"{order_domain}/global", item[0])
    )
    calibration_ids = {task_id for task_id, _ in global_order[:calibration_task_count]}
    calibration_specs = [item for item in specs if item[0] in calibration_ids]
    train_specs = [item for item in specs if item[0] not in calibration_ids]
    train_tasks = _materialize(
        train_specs, cohort="TRAIN", order_domain=f"{order_domain}/train", ordinal_offset=0
    )
    calibration_tasks = _materialize(
        calibration_specs,
        cohort="CALIBRATION",
        order_domain=f"{order_domain}/calibration",
        ordinal_offset=len(train_tasks),
    )

    def _record(task: QATask) -> dict[str, object]:
        return {
            "qa_id": task.qa_id,
            "cohort": task.cohort,
            "ordinal": task.ordinal,
            "references": list(task.references),
            "context": list(task.context),
            "question": task.question,
            "answer": task.answer,
        }

    curriculum_digest = sha256(
        _canonical_json(
            {
                "schema_version": QA_SCHEMA_VERSION,
                "corpus_source": CORPUS_SOURCE,
                "corpus_sha256": corpus_digest(),
                "seed": seed,
                "identifier_domain": identifier_domain,
                "order_domain": order_domain,
                "train": [_record(task) for task in train_tasks],
                "calibration": [_record(task) for task in calibration_tasks],
            }
        )
    ).hexdigest()
    return {
        "schema_version": QA_SCHEMA_VERSION,
        "corpus_source": CORPUS_SOURCE,
        "fixture_notice": FIXTURE_NOTICE,
        "corpus_size": len(entries),
        "corpus_sha256": corpus_digest(),
        "seed": seed,
        "identifier_domain": identifier_domain,
        "order_domain": order_domain,
        "train_tasks": tuple(train_tasks),
        "calibration_tasks": tuple(calibration_tasks),
        "sha256": curriculum_digest,
    }
