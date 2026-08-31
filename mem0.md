# Human Brain memory policy

Mem0 is project tooling for Codex sessions. It is not part of the Human Brain
runtime or evidence-producing model.

## Instructions

Remember durable repository-specific knowledge: architecture and dependency
decisions, scientific claim boundaries, accepted coding and testing conventions,
reproducible benchmark findings with their evidence paths, confirmed failure
modes, and approaches rejected with a reason.

Keep hypotheses visibly labeled as hypotheses. When evidence supersedes an older
memory, update the older memory or record that it is stale. Preserve provenance
for scientific results.

Ignore credentials, tokens, personal data, raw conversation transcripts,
transient shell output, stack traces, temporary file locations, generated
artifacts that are already reproducible from source, and speculative conclusions
presented without their uncertainty.

## Scope

Use the Mem0 application scope `human-brain` for every project memory. Do not mix
these memories with other repositories or global user preferences.

## Local setup

The repository's `.codex/config.toml` connects Codex directly to Mem0's hosted
MCP server. `MEM0_API_KEY` is stored in the Windows user environment; fully quit
and relaunch Codex after changing it, then verify a new task can list or search
Mem0 entities. The key must never be written to this repository.

This project intentionally uses Direct MCP. Do not also install the Mem0 plugin
unless the direct `[mcp_servers.mem0]` registration is removed first, or Codex
will register the same server twice.
