"""Command-line smoke test for the human-cell language model."""

from __future__ import annotations

import argparse

from humanbrain.language import CellLanguageModel


DEFAULT_CORPUS = (
    "Human cortical cells predict the next token.\n"
    "Dendritic branches remember context and pyramidal cells emit words.\n"
    "Human cortical cells grow connections when tokens repeat.\n"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Human cortical cells ")
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args(argv)
    model = CellLanguageModel.grow([DEFAULT_CORPUS])
    result = model.measure_completion(args.prompt, max_new_tokens=args.tokens)
    print(result["text"])
    print(f"{result['tokens_per_second']:.1f} simulated tokens/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

