# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""Plain Dagger SDK program (no module). Usage: dagger run python ci/main.py <rust|lint|test|smoke|all>.

Env knobs (all optional; unset locally):
  CI_CACHE_DIR    host dir used to persist the CACHES below between runs (RWX tool-cache)
  CI_CACHE_WRITE  "true" to export those caches back to CI_CACHE_DIR after the run
"""

import os
import sys
from pathlib import Path

import anyio
import dagger
from dagger import dag

REPO = Path(__file__).resolve().parent.parent
# Pins mirror .github/workflows/ci.yml (and uv in .rwx/tests.yml); bump them together.
RUST_IMAGE = "rust:1.99.1-bookworm"
NODE_IMAGE = "node:22.23.3-bookworm-slim"
UV_IMAGE = "ghcr.io/astral-sh/uv:0.13.5"
PYTHON = "3.12"  # ubuntu-24.04 system python, as ci.yml uses; ty fails on 3.11
UV_CACHE = "/root/.cache/uv"
CACHE_OUT = "/cache-out"
# Persisted download caches: seed name -> (mount path, Dagger cache volume).
# `npm ci` always recreates node_modules, so the npm cache is what pays off;
# Rust target/ is not persisted (see ci/README.md).
CACHES = {
    "cargo": ("/usr/local/cargo/registry", "xy-cargo-registry"),
    "uv": (UV_CACHE, "xy-uv"),
    "npm": ("/root/.npm", "xy-npm"),
    "playwright": ("/root/.cache/ms-playwright", "xy-playwright"),
    "pre-commit": ("/root/.cache/pre-commit", "xy-pre-commit"),
}
CHROMIUM = "$(node -e \"console.log(require('playwright').chromium.executablePath())\")"


def source() -> dagger.Directory:
    # .git is excluded on purpose: the version falls back to 0.0.0
    # (tool.uv-dynamic-versioning.fallback-version), and the upload stays small.
    return dag.host().directory(
        str(REPO),
        exclude=[
            ".git",
            ".context",
            ".venv",
            "**/.venv",
            "**/node_modules",
            "target",
            "dist",
            "**/__pycache__",
            "**/.pytest_cache",
            "**/.ruff_cache",
            "python/xy/static/index.js",
            "python/xy/static/standalone.js",
        ],
    )


def seed_dir(name: str) -> dagger.Directory | None:
    cache_dir = os.environ.get("CI_CACHE_DIR")
    path = Path(cache_dir) / name if cache_dir else None
    return dag.host().directory(str(path)) if path and path.is_dir() else None


def base() -> dagger.Container:
    node = dag.container().from_(NODE_IMAGE)
    ctr = (
        dag.container()
        .from_(RUST_IMAGE)
        .with_directory("/usr/local/node", node.directory("/usr/local"))
        .with_env_variable("PATH", "/usr/local/node/bin:${PATH}", expand=True)
        .with_file("/usr/local/bin/uv", dag.container().from_(UV_IMAGE).file("/uv"))
    )
    for name, (path, volume) in CACHES.items():
        ctr = ctr.with_mounted_cache(path, dag.cache_volume(volume), source=seed_dir(name))
    return (
        ctr.with_env_variable("UV_LINK_MODE", "copy")
        .with_env_variable("UV_PYTHON", PYTHON)
        .with_env_variable("UV_PYTHON_INSTALL_DIR", f"{UV_CACHE}/python")
        .with_env_variable("XY_REQUIRE_CARGO", "1")
        .with_directory("/src", source())
        .with_workdir("/src")
    )


def rust() -> dagger.Container:
    return (
        base()
        .with_exec(["rustup", "component", "add", "clippy"])
        .with_exec(["cargo", "test"])
        .with_exec(["cargo", "clippy", "--all-targets", "--", "-D", "warnings"])
    )


def synced() -> dagger.Container:
    # `uv sync` builds the project, which runs hatch_build.py (cargo + js/build.mjs).
    return (
        base()
        .with_exec(["npm", "ci"])
        .with_exec(["uv", "sync", "--locked", "--extra", "reflex", "--group", "dev"])
        .with_exec(["uv", "pip", "install", "-p", ".venv/bin/python", "matplotlib==3.11.0"])
    )


def git_indexed(ctr: dagger.Container) -> dagger.Container:
    # pre-commit and some tests call `git ls-files`; .git isn't uploaded, so
    # index the tree (honoring .gitignore) after the package build has run.
    return ctr.with_exec(["sh", "-c", "git init -q && git add -A"])


# `playwright install --with-deps` writes ~430 lines per check: apt-get output
# plus a download progress bar that Playwright always prints a line per 10%
# when stdout isn't a TTY (no flag turns it off). Keep it out of the CI log
# unless the install fails, then print all of it.
INSTALL_CHROMIUM = (
    'log="$(mktemp)"; '
    'npx playwright install --with-deps chromium >"$log" 2>&1 || { s=$?; cat "$log"; exit "$s"; }; '
    'grep "downloaded to" "$log" || echo "playwright: chromium already installed"'
)


def with_chromium(ctr: dagger.Container) -> dagger.Container:
    return ctr.with_exec(["sh", "-c", INSTALL_CHROMIUM]).with_exec(
        [
            "node",
            "--input-type=module",
            "-e",
            "import { chromium } from 'playwright';"
            "const b = await chromium.launch({executablePath: chromium.executablePath()});"
            "await b.close();",
        ]
    )


def lint() -> dagger.Container:
    checked = (
        synced()
        .with_exec([".venv/bin/python", "scripts/check_python_floor.py"])
        .with_exec([".venv/bin/python", "scripts/verify_ci_workflow.py"])
        .with_exec(
            [
                ".venv/bin/python",
                "-c",
                "import importlib.metadata as m, reflex_xy; "
                "assert reflex_xy.__version__ == m.version('xy')",
            ]
        )
        .with_exec([".venv/bin/ruff", "check", "."])
        .with_exec([".venv/bin/ruff", "format", "--check", "."])
        .with_exec([".venv/bin/python", "scripts/check_public_api.py"])
        .with_exec([".venv/bin/python", "scripts/check_typing.py"])
        # The repository hooks (ruff + docs codespell). Their entry points run
        # from the docs app's dev group; --only-group skips the Reflex site deps.
        .with_exec(["uv", "sync", "--project", "docs/app", "--frozen", "--only-group", "dev"])
    )
    return git_indexed(checked).with_exec(
        [
            "uv",
            "run",
            "--project",
            "docs/app",
            "--no-sync",
            "pre-commit",
            "run",
            "--all-files",
            "--show-diff-on-failure",
        ]
    )


def test() -> dagger.Container:
    return (
        with_chromium(git_indexed(synced()))
        .with_env_variable("XY_REQUIRE_BROWSER", "1")
        .with_exec(["sh", "-c", f'XY_CHROMIUM="{CHROMIUM}" exec .venv/bin/pytest -q'])
    )


# The `browser_smokes` job of ci.yml, minus its scope-gated pyplot pytest subset
# (already covered by `test`, which runs the whole suite with XY_REQUIRE_BROWSER).
# The stdlib smokes run under a package-free venv ($BARE), as ci.yml runs them
# before installing anything, so they keep proving the no-numpy/no-PyPI path.
BARE = "/tmp/bare/bin/python"
SMOKES = [
    "node -e \"import('./python/xy/static/index.js').then(m => { if (!m.default.render) throw new Error('no render export'); })\"",
    "node --check python/xy/static/standalone.js",
    "$BARE scripts/abi_smoke.py",
    '$BARE scripts/render_smoke_nonumpy.py "$CHROME"',
    '$BARE scripts/polar_parity_smoke.py "$CHROME"',
    '$BARE scripts/append_stream_smoke.py "$CHROME"',
    'XY_CHROMIUM="$CHROME" $BARE scripts/png_export_smoke.py',
    '.venv/bin/python scripts/smoke_render.py "$CHROME"',
    '.venv/bin/python scripts/polar_phase7_smoke.py "$CHROME"',
    '.venv/bin/python scripts/reflex_lifecycle_smoke.py "$CHROME"',
    '.venv/bin/python scripts/visual_regression_smoke.py "$CHROME"',
    '.venv/bin/python scripts/step_tier_smoke.py "$CHROME"',
    '.venv/bin/python scripts/interaction_stress_smoke.py "$CHROME"',
    ".venv/bin/python benchmarks/bench_dashboard.py --chart-counts 10,20,50,60"
    ' --chromium "$CHROME" --json /tmp/dashboard-smoke.json',
    ".venv/bin/python scripts/verify_benchmark_report.py /tmp/dashboard-smoke.json --kind dashboard-browser",
]


def smoke() -> dagger.Container:
    ctr = with_chromium(synced()).with_exec(["uv", "venv", "-q", "/tmp/bare"])
    for cmd in SMOKES:
        ctr = ctr.with_exec(["sh", "-c", f'set -e; BARE={BARE}; CHROME="{CHROMIUM}"; {cmd}'])
    return ctr


CHECKS = {"rust": rust, "lint": lint, "test": test, "smoke": smoke}


async def export_caches() -> None:
    # A fresh container mounting the same cache volumes, so a failed check
    # still persists what it downloaded.
    cache_dir = os.environ["CI_CACHE_DIR"]
    copies = " && ".join(f"cp -a {path} {CACHE_OUT}/{name}" for name, (path, _) in CACHES.items())
    ctr = base().with_exec(["sh", "-c", f"rm -rf {CACHE_OUT} && mkdir -p {CACHE_OUT} && {copies}"])
    await ctr.directory(CACHE_OUT).export(cache_dir)


async def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print("usage: main.py takes at most one check name", file=sys.stderr)
        return 2
    name = argv[0] if argv else "all"
    names = list(CHECKS) if name == "all" else [name]
    if any(n not in CHECKS for n in names):
        print(f"usage: main.py [{'|'.join([*CHECKS, 'all'])}]", file=sys.stderr)
        return 2
    write = os.environ.get("CI_CACHE_WRITE") == "true" and os.environ.get("CI_CACHE_DIR")
    async with dagger.connection(dagger.Config(log_output=sys.stderr)):
        try:
            for n in names:
                await CHECKS[n]().sync()  # a non-zero exit from any step raises here
        finally:
            if write:
                try:
                    await export_caches()
                except dagger.DaggerError as e:
                    # Must not displace the check failure being propagated.
                    print(f"cache export failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(anyio.run(main, sys.argv[1:]))
    except dagger.ExecError as e:
        sys.exit(e.exit_code or 1)
