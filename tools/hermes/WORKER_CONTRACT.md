# Hermes/Ox Human Brain worker contract

This contract governs external Hermes/Ox work on the Human Brain project. The
Kanban card is an orchestrator and its delegated children are untrusted patch
authors. Their reports and commits are proposals until a local reviewer checks
the diff and reruns the declared gates.

## Parent orchestrator

1. Verify that the worktree is clean and that `HEAD` equals every task's
   `base_sha`. Stop on any mismatch.
2. Read the attached prompt through EOF using line-offset pagination. For each
   line-safe record, concatenate `context_chunks` in order with no separator,
   then construct exactly `{goal, context, role: "leaf"}`. Pass all reconstructed
   objects to one `delegate_task(..., background=true)` call. Do not pass raw
   manifest rows, retain `context_chunks`, split the batch, add keys, or allow
   nested delegation.
3. Each fixed launcher-owned goal identifies only the validated task ID. Each
   reconstructed context contains this complete contract verbatim, the
   immutable run/routing pins, and that task's exact JSON record. A child has
   no access to the parent's conversation history.
4. On Hermes v0.20.5 a Kanban worker joins this requested-background batch
   synchronously and returns its consolidated results in the same parent turn.
   Capture every result before continuing; never issue a second batch or rely
   on a later goal turn to deliver a child result.
5. Synthesize the returned evidence into a manifest-hash-named report under
   `tools/hermes/reports/`. Record the manifest hash, base SHA, profile/model,
   parent worktree, every child branch and commit, files, tests, results, and
   limitations. Commit exactly that report and no other parent change, then
   include its path and commit SHA in `kanban_complete`.
6. Never merge, cherry-pick, rebase, push, or auto-approve a child branch.
   Child commits remain proposals for local review.

## Every child

- Before any source read, file tool, or terminal action, require the exact
  Hermes-injected `[WORKTREE ISOLATION]` suffix in the leaf goal with a concrete
  worktree path and dedicated branch. If it is absent, return `blocked`
  immediately. If present, verify that the current directory equals that path,
  the current branch equals the suffix branch, `HEAD` equals the task's exact
  `base_sha`, and the checkout is clean; block before source access on any
  mismatch.
- Read or modify only `allowed_paths`, plus this contract and root public
  project metadata needed to run the declared checks. Ask the parent to reject
  the task if its path allowance is insufficient.
- Use public source only. Never inspect sibling projects, `.runtime`, model
  caches, private bundles, hidden/canonical evaluation data, evaluation seeds,
  or prior private evaluation logs. Do not infer their contents from names or
  metadata.
- Respect `no_hidden_eval`: use public train/development material and
  independently constructed tests. Do not tune against held-out results, retry
  against hidden outcomes, or select among patches using hidden outcomes.
- Preserve the cell-only architecture. A tokenizer, compiler, strict syntax
  grammar, optimizer, and sandbox may support the cell simulation, but no LLM,
  Transformer, MLP, SSM, conventional solver, or hidden heuristic may select
  algorithms or generated programs.
- Stay within the task's `resource_class`. Only the one task explicitly marked
  `gpu` may reserve or benchmark the local GPU. `multimodal` permits only
  explicitly listed public inputs; it grants no broader filesystem access.
- Do not read credentials, alter Hermes/profile/global configuration, change
  approval policy, dispatch Kanban work, contact people, publish artifacts, or
  push code.
- Commit a coherent proposal on the child branch only when requested by the
  task. Never integrate it into another branch.

## Required child result

Return a compact structured report with:

- task ID and outcome (`complete`, `blocked`, or `no-change`);
- observed base SHA, final SHA, branch, and worktree;
- files read and files changed, each within `allowed_paths`;
- exact commands/checks run and their results;
- measured performance/capability evidence, with active/materialized/addressable
  cell counts distinguished when relevant;
- primary-source URLs for external factual claims;
- limitations, disconfirming evidence, and overfitting risks;
- the proposed commit SHA or patch description for local review.

A summary that claims success without reproducible evidence is incomplete.

## Security boundary

`allowed_paths` is an audited behavioral contract, not a filesystem sandbox.
Hermes worktrees isolate edits and branch collisions; they do not make sibling
or host files confidential. Source-upload authorization therefore assumes a
trusted local host and trusted agent runtime. Stop rather than dispatch if that
boundary is unsuitable for the manifest.
