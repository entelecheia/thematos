# Review instructions

Read by the required OCR delegation reviewer (Claude Code, Codex, or Kimi Code), additional `/codex:review` or `/code-review` passes, and human reviewers alike.

## Passes

Run these passes and tag every finding with its pass:

- Bugs: logic errors, broken edge cases, regressions.
- Security: HyFI configs choose which Python callables run. Look for config values that reach `eval`, `importlib`, or shell calls without going through `hyfi.instantiate` and its allowlist.
- Compliance: the change matches issue #85 (spec and decision table). Workflow changes (`.github/workflows/`) are checked against the workspace CI standard: every job has `timeout-minutes`, PR runs cancel in progress, docs-only changes skip the test matrix but still run the docs build, actions are pinned by SHA.
- Instruction prose (AGENTS.md, REVIEW.md): does each directive name the failure it prevents? Nits unless runtime behavior changes.

## Repo focus

- Topic model results are the product. Any change to `tests/data/metrics_1x.json`, to the tolerances in `tests/test_metrics_1x.py`, or to corpus building, word priors, or training order that moves those metrics without a new pinned test and a stated reason, is Important.
- No `hyfi`, `hydra`, or `omegaconf` import under `src/thematos/` (`tests/test_no_hyfi.py`).
- Every config node that builds an object has a `_target_` under `thematos.*`, or `lexikanon.*` for stopwords; a new target outside those needs a line in the allowlist note in `docs/usage.md`.
- Artifacts are written only under the model's `output_dir` (or an explicit `output_file`); a write anywhere else is Important.
- Known exception: lexikanon's `Stopwords.stopwords_fn` evaluates a string as Python. A change that lets thematos configs or data set that string is Important.

## Findings

Each finding carries its pass, severity, evidence (`file:line` or a reproduction), and provenance: introduced by this change, pre-existing, or indeterminate. Pre-existing findings go to a follow-up issue instead of widening the change. Report what was reviewed and what was not reached.

## Re-review

Hand the reviewer the diff, the spec, and this file, and let it read the rest of the repo (AGENTS.md included). Do not hand it fix-status claims, earlier dispositions, or do-not-reflag notes.

## Cap the nits

Report at most 5 nits per review; summarize the rest as a count.

## Do not report

- Generated paths: `uv.lock` (style only), `CHANGELOG.md`, `src/thematos/_version.py`.
- Binary test assets under `tests/assets/`.
- Anything CI already enforces: `uv lock --locked`, ruff check and format, mypy, pytest with the coverage floor, strict mkdocs build.
