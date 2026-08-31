# Human Brain agent instructions

## Mem0 project memory

Use Mem0 only as external development context. Keep it outside the cell models,
native simulation, frozen evaluation paths, and scientific evidence pipeline.
Any result influenced by retrieved Mem0 content is memory-augmented and is not
cell-only evidence.

Before work where earlier decisions, conventions, failed approaches, or measured
results could matter, search Mem0 with a task-specific query and
`filters={"app_id": "human-brain"}`. Read [mem0.md](mem0.md) before writing a
memory.

After significant verified work, store a short durable fact with
`app_id="human-brain"`. Include the source path or commit when it matters and
classify the memory as a decision, convention, anti-pattern, benchmark result,
or tooling setup. Prefer updating a contradicted memory over adding a competing
fact.

If Mem0 is unavailable or unauthenticated, continue the repository task and
report the memory setup issue. Repository work must not depend on Mem0 being
online.

## Codebase knowledge graph

Use `C-Users-laure-Desktop-DeepSim-Human-Brain` as the canonical
codebase-memory project for this checkout. After `list_projects`, verify that
its `root_path` resolves to the current repository before using graph results.
When a manual refresh is needed, reindex the exact checkout without a `name`
override so the index identity stays aligned with the session watcher.
