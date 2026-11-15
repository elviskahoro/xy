# CI (Dagger SDK program, no module)

`ci/main.py` is a plain Python program using `dagger-io` (pinned `0.21.8`, same line as the
Dagger CLI/engine `0.21.8`). RWX prepares tools and calls it; the same command runs locally.

## Run locally
Prereqs: Docker, `dagger` CLI 0.21.8, `uv`.

    ci/run.sh rust    # cargo test + clippy -D warnings
    ci/run.sh lint    # py3.11 floor syntax, CI workflow gate wiring, reflex_xy import, ruff, public API, ty,
                      # pre-commit --all-files (ruff hooks + docs codespell)
    ci/run.sh test    # full pytest with Playwright Chromium (XY_REQUIRE_BROWSER=1)
    ci/run.sh smoke   # browser_smokes job: JS parse, stdlib ABI/render/polar/append/PNG smokes (package-free
                      # venv), then numpy/Reflex render, lifecycle, visual, step-tier, stress, dashboard smokes
    ci/run.sh all     # rust, lint, test, smoke in order; stops at the first failure

`CI_CACHE_DIR` / `CI_CACHE_WRITE` are unset locally; Dagger uses its normal engine cache.

## Layers
| Layer | Where | What |
|---|---|---|
| A (RWX cached) | `.rwx/tests.yml` | `code`, `uv-install`, `dagger-cli-install` (sha256-pinned), `dagger-engine-warm`, `dagger-sdk-install` (filtered to `ci/pyproject.toml` + `ci/uv.lock`) |
| B (GitHub Actions, unchanged) | `.github/workflows/` | scope selection (`scripts/ci_scope.py`), shared wheel, pyplot gallery shards, regression/benchmark gates, conformance, release/docs workflows; still runs every job C mirrors |
| C (Dagger) | `ci/main.py` | `rust`, `lint`, `test`, `smoke` — the `rust_checks`, `test` and `browser_smokes` jobs of `ci.yml`, plus the repo pre-commit hooks |

Every check builds the native core from source (`uv sync` runs `hatch_build.py` → `cargo build --release`
and `node js/build.mjs`) and runs with `XY_REQUIRE_CARGO=1`, so a run never falls back to a missing core.
`test` sets `XY_REQUIRE_BROWSER=1`: the headless-Chromium suites (e.g. `tests/test_shared_glhost.py`) fail
rather than skip when Chromium can't launch. The `smoke` check leaves out the scope-gated pyplot pytest subset
of `browser_smokes`, because `test` already runs those files with a required browser.

## Relationship to GitHub Actions
RWX complements `ci.yml`; it doesn't replace it. `ci.yml` stays authoritative for required gates, scope
selection and everything in layer B, and the four release workflows stay owned by `reflex-release`.
`rwx-tests` gives a single pass/fail for the native core, lint, full pytest and browser suites, so a
sandbox with no Rust toolchain can verify changes end to end. Removing the mirrored `ci.yml` jobs is a
separate decision. It needs `rwx-tests` to be a required check that has stayed green on `main`.

## Caching
RWX is the only cache. The `checks` task has `docker: true`, `timeout: 90m`, `tool-cache: xy-checks`
(vault `xy-main`: PRs read, only main/`cache-rebuild` write via `CI_CACHE_WRITE=true`). It persists the
`CACHES` in `ci/main.py` under `$CI_CACHE_DIR`: the cargo registry, uv, npm (`npm ci` always recreates
`node_modules`, so the download cache is the part worth keeping), Playwright browsers and pre-commit hook
environments. Rust `target/` is not persisted: `hatch_build.py` expects `target/release`, the crate has two
small dependencies, and the fat-LTO link of the core itself dominates the build, so a multi-hundred-MB
tool-cache export would save little.
Expected hits: docs-only change → only `code` + `checks`; source-only → same; `ci/uv.lock` →
also `dagger-sdk-install`; Dagger bump → CLI/engine tasks.

## Notes
- `.git` is not uploaded to Dagger; the package version falls back to `0.0.0`.
- Dagger cannot run in Conductor cloud sandboxes (no Docker): validate with `rwx run .rwx/tests.yml --wait`.
- Pins (Rust 1.99.1, Node 22.23.3, uv 0.13.5, Python 3.12, the runner default) mirror `.github/workflows/ci.yml`.
