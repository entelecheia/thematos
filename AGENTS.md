# thematos

LDA topic modeling on tomotopy as a plain Pydantic library (`src/thematos/`), plus HyFI 2.0 config groups and workflows in `src/thematos/conf/` registered through the `hyfi.plugins` entry point. Workflow steps live in `src/thematos/steps.py`.

Design and decisions: issue #85 (the spec). Change the spec before the code when they disagree.

## Commands

- Install: `make install` (uv sync with all dependency groups)
- Lint: `make lint` (`uv lock --locked`, ruff check, ruff format --check, mypy on `src/thematos`)
- Test: `make test` (pytest with coverage, fails under the floor in `pyproject.toml`)
- Docs: `make docs-test` (strict mkdocs build)

## Verifying your work

```bash
make verify
```

Healthy output ends with `Required test coverage of 85.0% reached` and `N passed` (one AC3 test is skipped off macOS arm64). Run it before reporting any change complete and paste the tail. Fix the code when a test fails, never the test, its data, or the threshold.

## Conventions

- Python `>=3.11`; ruff targets py311, line length 100. Development pins 3.13 (`.python-version`).
- Library modules never import `hyfi`, `hydra`, or `omegaconf`; HyFI wiring lives only in `conf/*.yaml`. Enforced by `tests/test_no_hyfi.py`.
- Config files and class defaults must agree (`tests/test_hyfi_plugin.py::test_config_defaults_match_class_defaults`). Change both together.
- `tests/data/metrics_1x.json` is the recorded 0.14.5 result. Never regenerate it; an intended metric change gets its own test and a stated reason.
- Workflow examples in `conf/workflow/` are prefixed `thematos_` so they do not collide with other plugins.
- Conventional commits; python-semantic-release versions from tags on the `release` branch. Never hand-edit `src/thematos/_version.py` or the version in `pyproject.toml`.
- `uv.lock` is committed; re-run `uv lock` after editing dependencies or `make lint` fails.

## Things the agent gets wrong

- 2026-10-19, tomotopy keyword rename: 0.14 renamed `train(iter=)` / `infer(iter=)` to `iterations=`. Pass the count positionally to `train` and as `iterations=` to `infer`. Revisit if tomotopy renames them again.
- 2026-10-19, reproducibility: tomotopy results depend on the worker count. Tests and reference runs use `seed` plus `num_workers=1`. Revisit if tomotopy makes multi-worker training deterministic.
- 2026-10-19, manifest paths: HyFI records `project.workspace` as absolute, but values interpolated from it (`model.output_dir`) stay as resolved at compose time, relative to the run's cwd. Do not assert absolute `output_dir` in manifests. Revisit when HyFI absolutizes derived paths.
- 2026-10-19, fugashi has no cp315 wheel (via lexikanon): Python 3.15 builds it from source and needs `libmecab-dev` (CI installs it). Do not drop 3.15 from the matrix because a local build fails. Revisit when fugashi ships cp315 wheels.

---

Contract and update rule for this file: `_meta/rules/development-lifecycle.md` §6 in the workspace repo. When the same mistake appears twice, add the correction here in the same PR.
