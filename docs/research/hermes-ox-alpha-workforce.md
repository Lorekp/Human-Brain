# Hermes Agent and Ox Alpha as a parallel research workforce

**Date:** 2026-08-21

**Status:** First-party source audit plus three minimal live smoke tests. No credential value was read or printed, no Hermes configuration was changed, and no hidden or canonical simple-SWE evaluation artifact was opened or executed.

**Code baseline:** `06ad377fb860cb1011783d9cb2308ba03388495b`

**Scope boundary:** This note concerns using the locally installed Hermes Agent and the Nous Portal `stealth/ox-alpha` route for public research and isolated implementation work on `Human Brain`. It does not concern `Fly Brain`. It does not evaluate Ox Alpha's software-engineering quality, alter the cell-only model contract, or authorize external workers to inspect hidden evaluation data.

## Decision

No custom provider implementation is needed. The installed official Hermes Agent already supports Nous Portal OAuth, the exact Ox Alpha model ID, multimodal image ingestion, native tool calling, background subagent batches, nested orchestrators, live fleet monitoring, and per-child git-worktree isolation.

Use `stealth/ox-alpha` through provider `nous`. Keep the existing parent/planner selection independent, and route bounded delegated workers to Ox Alpha. Before allowing code-writing fan-out, enable `delegation.worktree_isolation`; it is currently unset and therefore defaults to `false`. Begin with the installed concurrency of 10, audit the first batch, then increase deliberately. The implementation has no numeric ceiling on `max_concurrent_children`, but the default per-turn loop guard permits only 50 spawned subagents and neither Nous nor Hermes publishes a per-account capacity guarantee for this route.

Treat the reported “one quadrillion tokens per day,” “limited time,” and comparisons with GLM 5.3 or GPT-5.6 Sol as unverified announcement/user claims until Nous publishes a durable first-party quota, end date, or benchmark. The official contract currently establishes zero token prices and route capabilities, not a service-level capacity allocation or comparative score.

## Claim ledger

| Claim | Finding | Evidence and boundary |
|---|---|---|
| Ox Alpha is available through Nous Portal | **Verified live** | The unauthenticated official [`/v1/models`](https://inference-api.nousresearch.com/v1/models) catalog returned `stealth/ox-alpha` on 2026-08-21, and a locally authenticated request completed successfully. The official Hermes commit that added it says it was verified against that live catalog ([commit `67af79d`](https://github.com/NousResearch/hermes-agent/commit/67af79d7e1bd8264119c2f02b37a2a0686008666)). |
| Exact Nous model identifier | **Verified:** `stealth/ox-alpha` | The live catalog's `id`, `canonical_slug`, and only alias all have that value. Do not use the sibling-provider IDs `x-preview-f-free` or `ox-alpha-free` for Nous. |
| Free via Nous | **Verified now** | Live pricing was `prompt: "0.0000000000"`, `completion: "0.0000000000"`; the official addition commit describes `$0/$0` on Portal. Zero current price is not a promise that the route remains free. |
| One quadrillion tokens/day | **Not independently verified** | No such quota appears in the official live model entry, relevant official Hermes source/history, public Hermes documentation, or response rate-limit headers. `per_request_limits` is `null`; that means the catalog publishes no per-request limit, not that usage is unlimited. The quoted number may describe aggregate provider capacity rather than a guaranteed per-account allocation; that is an inference from the wording, not a verified fact. |
| Free for a limited time | **Not independently bounded** | The official route is free now, but no public first-party end date or revocation condition was found. The catalog's `expiration_date: "2098-12-31"` should be treated as catalog metadata/sentinel, not as a free-price guarantee through 2098. |
| Comparable to GLM 5.3 / Sol 5.6 max | **Not verified** | The official API describes long-horizon coding and agent work but publishes no comparative benchmark in the sources reviewed. The smoke tests below establish connectivity and features only, not model quality. |
| Multimodal | **Verified route contract** | The live catalog reports `text+image+video->text`, with text, image, and video inputs and text output. It does not advertise audio or PDF input. A local `hermes chat --image` smoke test also completed, but Hermes preprocesses CLI attachments through its auxiliary vision path; the live Nous catalog is the evidence for the model route's native modalities. |
| Tool calling | **Verified contract and live execution** | The catalog advertises `tools` and `tool_choice`; a live Ox Alpha Hermes run used the terminal tool and returned the exact worktree HEAD. |
| Very large context | **Verified** | Context length is 1,048,576 tokens and maximum completion length is 131,072 tokens in the live catalog and official addition commit. |
| Reasoning controls | **Verified** | Reasoning is mandatory and enabled by default; accepted efforts are `low`, `high`, and `max`, with `max` the route default. Installed Hermes maps its user-facing `xhigh` choice to `max` for Ox Alpha ([source](https://github.com/NousResearch/hermes-agent/blob/ac8dff4fbcf47a392a3cddcbec068aa05930ab47/agent/reasoning_effort.py#L98-L103)). |

## Installed state on this machine

The executable is:

```text
C:\Users\laure\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe
```

Read-only inspection produced:

```text
Hermes Agent v0.20.5 (2026.8.19)
Install method: git
Python: 3.11.15
OpenAI SDK: 2.24.0
Official origin: https://github.com/NousResearch/hermes-agent.git
Installed source: ac8dff4fbcf47a392a3cddcbec068aa05930ab47
```

`hermes portal info` reported that the profile is logged in to `https://portal.nousresearch.com` and uses `https://inference-api.nousresearch.com/v1`, without displaying a credential. It also reported that the active main provider remains OpenRouter. No login step is required for a per-invocation Ox call on this profile.

The relevant resolved configuration, read one scalar at a time with `hermes config get`, is:

| Setting | Current value | Consequence |
|---|---:|---|
| `model.provider` | `openrouter` | The ordinary parent session does not use Nous unless overridden. |
| `model.default` | `deepseek/deepseek-v4-pro-0813` | Ox is not the default main model. |
| `delegation.provider` | `nous` | Delegated children already route through Nous. |
| `delegation.model` | `deepseek/deepseek-v4-flash-0731` | Delegated children do not yet use Ox. |
| `delegation.reasoning_effort` | `max` | Compatible with Ox. |
| `delegation.max_concurrent_children` | `10` | At most ten children run concurrently in one batch. |
| `delegation.max_iterations` | `500` | Children have a generous tool-turn budget; prompts still need narrow termination criteria. |
| `delegation.max_spawn_depth` | `1` | Delegation is flat; children cannot recursively spawn workers. |
| `delegation.subagent_auto_approve` | `false` | Dangerous child commands are denied non-interactively, the safe setting. |
| `delegation.worktree_isolation` | unset (`false` by default) | Code-writing children would share the checkout and can collide. |
| `tool_loop_guardrails.loop_caps.max_subagents` | `50` | One parent turn may spawn at most 50 children, counting every task in a batch. |
| `max_concurrent_sessions` | `null` | No configured gateway-wide session cap; this is not a provider quota. |

The official hosted delegation page still says “3 concurrent subagents by default,” but installed `v0.20.5` source sets the default to **10**, and this profile resolves to 10. For reproducibility, prefer `hermes config get delegation.max_concurrent_children` and the pinned installed source over the stale prose default ([installed-source lines](https://github.com/NousResearch/hermes-agent/blob/ac8dff4fbcf47a392a3cddcbec068aa05930ab47/tools/delegate_tool.py#L846-L889), [hosted delegation documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation)).

`hermes portal tools` reported that Portal-managed web search/extraction, image generation, TTS, browser automation, and related Tool Gateway services are not configured. That is separate from both the locally enabled terminal/file/delegation tools and Ox Alpha's native image/video input. The [Tool Gateway documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway) explains those optional managed services.

## Official Ox Alpha route contract

The following projection is reproducible without an API key because the official catalog exposes model metadata publicly:

```powershell
$catalog = Invoke-RestMethod `
  -Uri 'https://inference-api.nousresearch.com/v1/models' `
  -Headers @{Accept='application/json'; 'User-Agent'='Hermes-Agent/0.20.5'}
$catalog.data |
  Where-Object { $_.id -eq 'stealth/ox-alpha' } |
  ConvertTo-Json -Depth 12
```

Observed contract on 2026-08-21:

```text
id/canonical slug:       stealth/ox-alpha
input modalities:       text, image, video
output modalities:      text
context length:         1,048,576 tokens
max completion:         131,072 tokens
prompt price:           0
completion price:       0
per-request limits:     not published (null)
reasoning:              mandatory; low/high/max; default max
tool parameters:        tools, tool_choice
structured response:    response_format
other controls:         max_tokens, temperature, top_k, top_p,
                        include_reasoning, reasoning, reasoning_effort
moderation flag:        false
```

The absence of a moderation flag is API metadata, not permission to disregard project policy, data licenses, confidentiality, or evaluator isolation.

## Setup and invocation

### Authentication

For a new profile, use the official OAuth flow; do not paste a Portal token into scripts:

```powershell
hermes portal login
# Equivalent lower-level form:
hermes auth add nous --type oauth
```

The installed CLI defaults to `https://portal.nousresearch.com`, inference at `https://inference-api.nousresearch.com/v1`, and OAuth client ID `hermes-cli` ([official source](https://github.com/NousResearch/hermes-agent/blob/ac8dff4fbcf47a392a3cddcbec068aa05930ab47/hermes_cli/auth.py#L112-L114)). Verify only redacted state:

```powershell
hermes portal info
hermes auth status nous
```

### One bounded Ox worker without changing defaults

```powershell
hermes chat `
  --provider nous `
  --model stealth/ox-alpha `
  --reasoning max `
  --worktree `
  --in 'C:\Users\laure\Desktop\DeepSim\Human Brain' `
  --query '<complete bounded task, allowed paths, tests, and output contract>'
```

Top-level `--worktree` sessions normally branch from a freshly fetched upstream tip when one is available, whereas delegated `worktree_isolation` branches from the parent's current `HEAD`. If a task must use an exact local base SHA, either create the worktree explicitly first and pass it with `--in`, or set `worktree_sync: false` for that controlled run and verify `git rev-parse HEAD` before the agent edits anything.

Prefer `hermes chat --query` for code-writing automation. The shorter top-level `hermes -z/--oneshot` mode explicitly auto-bypasses approvals, and `--yolo` also bypasses dangerous-command prompts; neither belongs in the default code fleet.

### Route delegated workers to Ox Alpha

These are the minimal persistent changes the operator or parent tooling task may choose to apply; this research task did **not** apply them:

```powershell
hermes config set delegation.provider nous
hermes config set delegation.model stealth/ox-alpha
hermes config set delegation.reasoning_effort max
hermes config set delegation.worktree_isolation true
hermes config check
```

Leaving `model.provider` and `model.default` unchanged keeps the existing parent/planner while moving the high-volume delegated work to Ox. To make Ox the main interactive model too, the supported user flow is `hermes model`; a scripted equivalent is:

```powershell
hermes config set model.provider nous
hermes config set model.default stealth/ox-alpha
hermes config check
```

Use per-invocation overrides until the route has passed a representative pilot. They avoid surprising unrelated Hermes sessions.

## Parallel workflow mechanics and limits

Hermes's `delegate_task(tasks=[...])` creates fresh child agents with isolated conversation contexts, inherited tool access, and separate terminal sessions. A top-level batch runs in the background and returns a handle; `/agents` or `/tasks` displays live branches, cost/token/file rollups, and pause/kill controls. Only final child summaries enter the parent context. The official [delegation documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation) is the primary operational reference.

Important limits and failure modes:

1. **Concurrency is configurable, not proven infinite.** Installed source floors `delegation.max_concurrent_children` at 1 and has no numeric ceiling. Each child still consumes a provider request, Python thread, terminal, memory, network socket, and potentially compiler/GPU resources. Hundreds of configured slots are therefore not evidence that hundreds will run productively.
2. **There is a separate per-turn spawn cap.** `tool_loop_guardrails.loop_caps.max_subagents` defaults to 50; a batch task counts as one spawn per child. Raising concurrency above 50 without also raising this guard cannot produce a flat 100-child batch.
3. **Nested fan-out is off.** `max_spawn_depth=1` is flat. An `orchestrator` child can spawn descendants only after the depth is raised. Branching multiplies load exponentially and makes ownership harder to audit.
4. **Filesystem isolation is opt-in and can degrade.** With delegated `worktree_isolation: true`, local git children receive `.worktrees/subagent-<id>` branches from the parent's current `HEAD`. On non-git directories, non-local terminal backends, or setup failure, Hermes silently falls back to a shared workspace. Check each result's `worktree` record and base SHA rather than assuming isolation. The separate top-level `hermes --worktree` path has the upstream-sync behavior noted above.
5. **Background completion is durable; execution is not.** A completed-but-undelivered result survives restart, but an in-flight child becomes `unknown`. Use cron or a supervised external process for jobs that must survive a Hermes restart.
6. **Subagents inherit credentials and toolsets.** Leaf agents cannot delegate, clarify, write shared memory, send messages, or schedule cron, but they retain terminal/file/code tools. Start the parent with the minimum toolsets needed.
7. **Free tokens do not remove review cost.** Large fan-out increases duplicated work, conflicting edits, flaky integration, host load, and the volume of untrusted patches. Review and tests remain the bottleneck.

For a flat 50-worker experiment, after the 10-worker pilot succeeds:

```powershell
hermes config set delegation.max_concurrent_children 50
hermes config set tool_loop_guardrails.loop_caps.max_subagents 50
```

For more than 50 children in one turn, both numbers must rise. Do not set either to `0` (unlimited) for this project. Increase 10 -> 25 -> 50 while recording success rate, first-token and completion latency, HTTP 429/5xx responses, host RAM/handle usage, duplicate-task rate, clean-worktree rate, test-pass rate, and useful accepted commits per million tokens. Only then test 100. There is no first-party evidence that the reported aggregate token capacity also guarantees 100 simultaneous requests from this account.

## Project-specific fleet protocol

Use Ox as a force multiplier around the cell simulation, never as a hidden replacement for it.

Every work packet should contain:

- the exact `Human Brain` base SHA and a single non-overlapping responsibility;
- explicit allowed write paths and the instruction not to touch `Fly Brain`;
- a prohibition on hidden/canonical simple-SWE seeds, bundles, completions, ledgers, or evaluator internals;
- the cell-only architectural contract: no Transformer, MLP, SSM, external solver, hidden test search/repair loop, or score-reordering shortcut in the product;
- required tests/benchmarks and a wall-clock/resource budget;
- an output contract: commit hash, changed paths, commands run, raw results, limitations, and a no-eval-access attestation; and
- a stop condition so a child does not broaden the task.

A useful initial 10-lane batch is:

1. two read-only primary-source reviews of human E/I dynamics and local learning;
2. two independent native/CUDA profiling and kernel-design reviews;
3. one procedural 64-bit topology review;
4. one sparse readout and decoder review;
5. one scale-law experiment-design review with preregistered comparisons;
6. one anti-overfit/evaluator-boundary adversarial review;
7. one test/benchmark reproducibility review; and
8. one synthesis agent that reads only the other summaries and proposes a ranked experiment queue.

Code-writing tasks get separate worktrees and distinct files. Research-only agents may share a read-only checkout. The parent reviews diffs and tests locally before cherry-picking. No external agent receives the canonical hidden evaluation seed or bundle, and no result is called a capability improvement until the local cell model reproduces it under the frozen public-development protocol.

## Live smoke tests

The following calls used provider/model overrides, did not change defaults, and ran outside the hidden evaluation harness.

### Text route

```powershell
hermes chat -q 'Return exactly OX_ALPHA_OK and nothing else.' `
  --provider nous -m stealth/ox-alpha --reasoning low `
  -Q --max-turns 2 --run-budget 180 --ignore-rules --source tool
```

Result: `OX_ALPHA_OK` with exit code 0.

### Native tool call

```powershell
hermes chat -q 'Use the terminal tool to run git rev-parse HEAD in the current directory. Return only the exact 40-character hash from the tool result.' `
  --provider nous -m stealth/ox-alpha --reasoning low -t terminal `
  -Q --max-turns 4 --run-budget 180 --ignore-rules --source tool
```

Result: `06ad377fb860cb1011783d9cb2308ba03388495b`, matching the isolated worktree. Hermes logged three model API calls and two tool turns.

### Image path

An official public `hermes-agent-banner.png` from the installed Nous repository was attached with `hermes chat --image`. The end-to-end command returned `HERMES AGENT`, matching the banner. Hermes's CLI first converts an attachment to an auxiliary vision description before the main conversation, so this proves the installed image workflow, not by itself that the main chat request carried raw image bytes. The official live catalog independently establishes that the Ox route accepts native image and video inputs.

The first smoke attempt from the restricted filesystem/network sandbox failed with local log/session-store permission errors and network connection retries. Running the same bounded command in the normal user context succeeded. This was an execution-context limitation, not evidence of a Nous route outage.

## What remains unverified

- the announcement's one-quadrillion-token daily capacity and whether it is global, organizational, or per account;
- a free-access end date, fair-use policy, request-per-minute limit, token-per-minute limit, or concurrency SLA;
- performance parity with GLM 5.3 or GPT-5.6 Sol on any published benchmark;
- quality under hundreds of parallel software agents;
- native video handling through the current Hermes CLI, whose direct single-query attachment flag is `--image`; and
- behavior under long context, 131K output, sustained rate limiting, or process restart.

These are reasons to instrument the pilot, not reasons to avoid Ox. The verified capability is already valuable: a live, zero-priced, million-token, multimodal, tool-calling reasoning route can supply many isolated research and implementation workers while the cell model, tests, and evaluator remain the source of truth.

## Primary sources

- [Official Nous inference model catalog](https://inference-api.nousresearch.com/v1/models) — live model ID, modalities, limits, pricing, parameters, and reasoning contract.
- [Official Hermes commit adding Ox Alpha to Nous Portal](https://github.com/NousResearch/hermes-agent/commit/67af79d7e1bd8264119c2f02b37a2a0686008666) — Portal addition and the maintainers' live-catalog verification.
- [Hermes Agent subagent delegation](https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation) — batching, monitoring, ownership, worktrees, nesting, and durability.
- [Hermes Agent configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) — loop guards, auxiliary vision, delegation, and session limits.
- [Nous Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway) — optional Portal-managed tools, distinct from model modalities and local built-ins.
- [Installed pinned source: delegation implementation](https://github.com/NousResearch/hermes-agent/blob/ac8dff4fbcf47a392a3cddcbec068aa05930ab47/tools/delegate_tool.py) and [configuration defaults](https://github.com/NousResearch/hermes-agent/blob/ac8dff4fbcf47a392a3cddcbec068aa05930ab47/hermes_cli/config_defaults.py) — current default 10-child concurrency, no numeric ceiling, spawn-depth semantics, and 50-child per-turn guard.
- [Installed pinned source: Ox reasoning effort](https://github.com/NousResearch/hermes-agent/blob/ac8dff4fbcf47a392a3cddcbec068aa05930ab47/agent/reasoning_effort.py#L98-L103) and [model metadata](https://github.com/NousResearch/hermes-agent/blob/ac8dff4fbcf47a392a3cddcbec068aa05930ab47/agent/model_metadata.py#L562-L567) — mandatory reasoning controls and context metadata.
