# CI (Dagger SDK program, no module)

`ci/main.py` is a plain Python program using `dagger-io` (pinned `0.21.8`, same line as the
Dagger CLI/engine `0.21.8`). RWX prepares tools and calls it; the same command runs locally.

## Run locally
Prereqs: Docker, `dagger` CLI 0.21.8, `uv`.

    ci/run.sh rust    # cargo test + clippy -D warnings
    ci/run.sh lint    # py3.11 floor syntax, CI workflow gate wiring, reflex_xy import, ruff, public API, ty
    ci/run.sh test    # pytest with Playwright Chromium (XY_REQUIRE_BROWSER=1)
    ci/run.sh all

`CI_CACHE_DIR` / `CI_CACHE_WRITE` are unset locally; Dagger uses its normal engine cache.

## Layers
| Layer | Where | What |
|---|---|---|
| A (RWX cached) | `.rwx/tests.yml` | `code`, `uv-install`, `dagger-cli-install` (sha256-pinned), `dagger-engine-warm`, `dagger-sdk-install` (filtered to `ci/pyproject.toml` + `ci/uv.lock`) |
| B (GitHub Actions, unchanged) | `.github/workflows/` | scope selection (`scripts/ci_scope.py`), shared wheel, pyplot gallery shards, browser smokes, regression/benchmark gates, conformance, release/docs workflows |
| C (Dagger) | `ci/main.py` | `rust`, `lint`, `test` — the `rust_checks` and `test` jobs of `ci.yml` |

## Caching
RWX is the only cache. The `checks` task has `docker: true`, `timeout: 60m`, `tool-cache: xy-checks`
(vault `xy-main`: PRs read, only main/`cache-rebuild` write via `CI_CACHE_WRITE=true`). It persists the
cargo registry and uv caches in `$CI_CACHE_DIR`. Rust `target/` is not persisted: `hatch_build.py`
expects `target/release`.
Expected hits: docs-only change → only `code` + `checks`; source-only → same; `ci/uv.lock` →
also `dagger-sdk-install`; Dagger bump → CLI/engine tasks.

## Notes
- `.git` is not uploaded to Dagger; the package version falls back to `0.0.0`.
- Dagger cannot run in Conductor cloud sandboxes (no Docker): validate with `rwx run .rwx/tests.yml --wait`.
- Pins (Rust 1.99.1, Node 22.23.3, uv 0.13.5, Python 3.12, the runner default) mirror `.github/workflows/ci.yml`.
