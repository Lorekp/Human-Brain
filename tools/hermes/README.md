# Hermes/Ox workforce launcher

`run_fleet.ps1` turns one reviewed JSONL manifest into one durable Hermes
Kanban goal card. That parent card launches one in-process background
`delegate_task` batch. Hermes v0.20.5 joins that requested-background batch
synchronously inside the Kanban worker, so the parent captures all child
results in the same turn, writes a hash-named committed report, and leaves
every child branch unmerged for local review.

The launcher is intentionally narrower than the Hermes CLI:

- fixed profile: `humanbrainox`;
- fixed dedicated board: `human-brain-ox`;
- fixed provider/model: `nous:stealth/ox-alpha`;
- at most 128 tasks and at most one `gpu` task;
- a clean worktree and one exact base SHA across all task rows;
- plan-only unless both `-Execute` and `-AuthorizeSourceUpload` are present;
- one blocked goal card, one joined delegate batch, one `dispatch --max 1`,
  one retry at most, and no integration action.

It is pinned to Hermes Agent v0.20.5. The profile, dedicated board, Portal
authentication, 128-child delegation limit, depth-two worktree isolation, and
approval policy are installation-owned prerequisites. This launcher reads but
never changes that global state. It routes the card through an explicit
`worktree:<validated repository>` workspace and a deterministic local branch
that is verified at the manifest base SHA; it does not trust mutable project
routing.

## Prepare and review a manifest

An executable manifest must live outside the repository: an untracked manifest
inside the worktree correctly fails the clean-checkout gate. Copy
`tasks.example.jsonl`, replace every placeholder with the same current `HEAD`,
then edit the bounded tasks. For example:

```powershell
$repo = "C:\Users\laure\Desktop\DeepSim\Human Brain"
$base = git -C $repo rev-parse HEAD
$manifest = Join-Path $env:TEMP "human-brain-ox-tasks.jsonl"
(Get-Content .\tools\hermes\tasks.example.jsonl -Raw).Replace(
    "REPLACE_WITH_40_CHARACTER_LOWERCASE_GIT_SHA",
    $base
) | Set-Content -LiteralPath $manifest -Encoding utf8NoBOM

.\tools\hermes\run_fleet.ps1 `
    -ManifestPath $manifest `
    -RepositoryPath $repo
```

The last command validates everything, prints the manifest hash, base SHA,
branch/idempotency identity, profile/model, and exact dry-run shape, but does
not resolve or invoke Hermes.

After reviewing that output, explicitly authorize both execution and the fact
that the external model may receive source from the allowed public paths:

```powershell
.\tools\hermes\run_fleet.ps1 `
    -ManifestPath $manifest `
    -RepositoryPath $repo `
    -Execute `
    -AuthorizeSourceUpload
```

The script writes the full parent prompt to a temporary UTF-8 file and creates
the card initially blocked. It attaches the prompt through Hermes' native
Kanban attachment command, verifies the stored file's identity and SHA-256,
then promotes the card only after proving the board has no other ready card.
Immediately before the board-wide `dispatch --max 1`, it rechecks the clean
base and pinned branch, proves the promoted card is the sole ready card, and
afterward verifies that Hermes spawned that exact card. No task text is
evaluated by a shell or placed on the Windows command line, and the temporary
file is removed. A non-archived deterministic fleet card makes a rerun fail;
inspect and archive it manually rather than replaying a delegated batch.

These checks narrow trusted-host races but cannot eliminate them if another
local process mutates Git refs or the board between CLI calls. Likewise,
`allowed_paths` is an audited worker obligation, not a hard read sandbox:
worktrees isolate writes and collisions, not confidentiality. Both execution
switches authorize sending the listed public source to that trusted runtime.

## JSONL task schema

Every nonblank line is exactly one object with these fields and no others:

- `id`: unique lowercase kebab-case identifier, at most 64 characters;
- `title` and `objective`: nonempty bounded strings;
- `base_sha`: the clean repository's exact 40-character lowercase `HEAD`;
- `allowed_paths`: nonempty canonical repo-relative public paths;
- `acceptance`: nonempty list of reproducible checks;
- `resource_class`: `cpu`, `gpu`, `research`, or `multimodal`;
- `public_source_only`: exactly `true`;
- `no_hidden_eval`: exactly `true`.

Sibling-project, private-runtime, model-cache, hidden-evaluation, evaluation-seed,
and traversal paths are rejected before Hermes command resolution. The worker
contract adds the runtime obligations and result format.
