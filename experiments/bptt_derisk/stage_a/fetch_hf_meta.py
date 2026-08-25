#!/usr/bin/env python3
"""Stage A provenance: dump verbatim HF API metadata for the three repos.

Reads the Hub API (blobs=true) for
  ridger/SpikeGPT-OpenWebText-216M,
  ridger/SpikeGPT-BookCorpus,
  Brain2nd/NeuronSpark-0.9B
and writes stage_a/hf_api_meta.json with exact per-file sizes as reported by
the Hub at fetch time. This is the integrity target the downloads are checked
against. No auth required (all three repos are public).
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPOS = [
    "ridger/SpikeGPT-OpenWebText-216M",
    "ridger/SpikeGPT-BookCorpus",
    "Brain2nd/NeuronSpark-0.9B",
]
OUT = Path(__file__).resolve().parent / "hf_api_meta.json"


def fetch(repo: str) -> dict:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "stage-a-provenance/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode("utf-8"))
    files = {}
    for s in d.get("siblings", []):
        name = s.get("rfilename")
        size = s.get("size")
        if name is not None and size is not None:
            files[name] = {"size": int(size), "blob_id": s.get("blobId") or s.get("lfs", {}).get("oid")}
    return {
        "repo": repo,
        "api_sha": d.get("sha"),
        "lastModified": d.get("lastModified"),
        "tags": d.get("tags"),
        "files": files,
    }


def main() -> None:
    out = {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repos": {}}
    for repo in REPOS:
        for attempt in range(3):
            try:
                out["repos"][repo] = fetch(repo)
                break
            except Exception as e:  # noqa: BLE001
                print(f"retry {repo} ({attempt}): {e}", flush=True)
                time.sleep(5)
        else:
            out["repos"][repo] = {"repo": repo, "error": "unreachable"}
        r = out["repos"][repo]
        total = sum(f["size"] for f in r.get("files", {}).values())
        print(f"{repo}: {len(r.get('files', {}))} files, {total:,} bytes total", flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
