"""Grow a recurrent human-cell checkpoint from a verified teacher trajectory."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from humanbrain.language import CellLanguageModel, ModelConfig
from humanbrain.swe_growth import growth_examples_from_trajectory
from humanbrain.swe_model import DEFAULT_ACTION_REGEX


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "swe-dev-expert"
            / "sqlfluff__sqlfluff-1625"
            / "teacher.traj.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "swe_sqlfluff_1625_cells.npz",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--recall-limit", type=int, default=None)
    parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="skip growth and run the bounded recall gate on an existing checkpoint",
    )
    args = parser.parse_args()

    trajectory_bytes = args.trajectory.read_bytes()
    trajectory = json.loads(trajectory_bytes)
    examples = growth_examples_from_trajectory(trajectory)
    config = ModelConfig(
        max_order=64,
        max_phrases=2048,
        min_phrase_frequency=2,
    )

    if args.reuse_checkpoint:
        model = CellLanguageModel.load(args.checkpoint)
        growth_seconds = 0.0
        checkpoint_sha256 = sha256(args.checkpoint.read_bytes()).hexdigest()
    else:
        started = time.perf_counter()
        model = CellLanguageModel.grow(
            [example.sequence for example in examples],
            config,
        )
        growth_seconds = time.perf_counter() - started
        checkpoint_sha256 = model.save(args.checkpoint)

    recall = []
    pattern = re.compile(DEFAULT_ACTION_REGEX, re.DOTALL)
    recall_examples = examples[: args.recall_limit]
    for index, example in enumerate(recall_examples, start=1):
        generated = model.complete(
            example.prompt,
            max_new_tokens=args.max_new_tokens,
            stop_text="</bash>",
        )
        matches = pattern.findall(generated)
        decoded_commands = []
        for match in matches:
            try:
                decoded_commands.append(json.loads(f'"{match}"'))
            except json.JSONDecodeError:
                decoded_commands.append("<invalid-json>")
        recall.append(
            {
                "step": index,
                "pass": decoded_commands == [example.command],
                "command_count": len(decoded_commands),
                "command_sha256": [
                    sha256(command.encode("utf-8")).hexdigest()
                    for command in decoded_commands
                ],
                "expected_command_sha256": sha256(
                    example.command.encode("utf-8")
                ).hexdigest(),
                "generated_sha256": sha256(generated.encode("utf-8")).hexdigest(),
                "generated_preview": generated[:240],
            }
        )

    output = {
        "pass": all(item["pass"] for item in recall),
        "trajectory": str(args.trajectory.resolve()),
        "trajectory_sha256": sha256(trajectory_bytes).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "growth_examples": len(examples),
        "growth_seconds": growth_seconds,
        "vocabulary_size": model.vocabulary_size,
        "grown_branches": model.grown_branches,
        "training_digest": model.training_digest,
        "max_new_tokens": args.max_new_tokens,
        "recall": recall,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
