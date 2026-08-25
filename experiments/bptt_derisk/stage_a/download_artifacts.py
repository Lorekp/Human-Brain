#!/usr/bin/env python3
"""Stage A downloads: SpikeGPT 216M x2 + NeuronSpark-0.9B (code+weights), free.

snapshot_download()s the three public repos into .models/stage_a/ (git-ignored),
then verifies every downloaded file against the Hub-reported byte sizes in
hf_api_meta.json and records sha256 for the weight files. Writes
stage_a/download_manifest.json (sizes, sha256, hf commit sha) and prints a
summary line per repo. No training, no spend.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

STAGE = Path(__file__).resolve().parent
REPO_ROOT = STAGE.parents[2]
CACHE = REPO_ROOT / ".models" / "stage_a"
META = json.loads((STAGE / "hf_api_meta.json").read_text(encoding="utf-8"))
REPOS = list(META["repos"].keys())
WEIGHT_SUFFIXES = (".pth", ".safetensors", ".bin")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repos": {}}
    for repo in REPOS:
        expected = META["repos"][repo].get("files", {})
        print(f"=== downloading {repo} -> {CACHE / repo.split('/')[-1]}", flush=True)
        t0 = time.time()
        local = snapshot_download(
            repo_id=repo,
            local_dir=str(CACHE / repo.split("/")[-1]),
            token=False,
            max_workers=4,
        )
        local = Path(local)
        rows, mismatches = [], []
        total_bytes = 0
        for rel, meta in sorted(expected.items()):
            p = local / rel
            if not p.exists():
                mismatches.append({"file": rel, "issue": "MISSING after download"})
                continue
            size = p.stat().st_size
            total_bytes += size
            row = {
                "file": rel,
                "expected_size": meta["size"],
                "actual_size": size,
                "size_match": size == meta["size"],
            }
            if not row["size_match"]:
                mismatches.append({"file": rel, "issue": f"size {size} != {meta['size']}"})
            if rel.endswith(WEIGHT_SUFFIXES):
                digest = sha256_file(p)
                row["sha256"] = digest
            rows.append(row)
        # any extra local files not in the API listing
        extras = [
            str(p.relative_to(local)).replace("\\", "/")
            for p in local.rglob("*")
            if p.is_file() and str(p.relative_to(local)).replace("\\", "/") not in expected
        ]
        manifest["repos"][repo] = {
            "local_dir": str(local),
            "hf_commit_sha": META["repos"][repo].get("api_sha"),
            "files": rows,
            "extras_not_in_api_listing": sorted(extras),
            "total_downloaded_bytes": total_bytes,
            "integrity_mismatches": mismatches,
            "download_wall_s": round(time.time() - t0, 1),
        }
        ok = "OK" if not mismatches else "MISMATCH"
        print(f"{repo}: {len(rows)} files verified [{ok}], {total_bytes:,} bytes, "
              f"{manifest['repos'][repo]['download_wall_s']}s", flush=True)
    out = STAGE / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
