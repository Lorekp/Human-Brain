"""Plan or explicitly execute the preregistered public typed TRAIN curriculum."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from humanbrain.typed_curriculum import run  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run())
