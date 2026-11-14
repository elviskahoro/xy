# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""Plain Dagger SDK program (no module). Usage: dagger run python ci/main.py <rust|lint|test|all>.

Env knobs (all optional; unset locally):
  CI_CACHE_DIR    host dir used to persist the cargo + uv caches between runs (RWX tool-cache)
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
CARGO_HOME = "/usr/local/cargo"
UV_CACHE = "/root/.cache/uv"
CACHE_OUT = "/cache-out"


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
    return (
        dag.container()
        .from_(RUST_IMAGE)
        .with_directory("/usr/local/node", node.directory("/usr/local"))
        .with_env_variable("PATH", "/usr/local/node/bin:${PATH}", expand=True)
        .with_file("/usr/local/bin/uv", dag.container().from_(UV_IMAGE).file("/uv"))
        .with_mounted_cache(
            f"{CARGO_HOME}/registry",
            dag.cache_volume("xy-cargo-registry"),
            source=seed_dir("cargo"),
        )
        .with_mounted_cache(UV_CACHE, dag.cache_volume("xy-uv"), source=seed_dir("uv"))
        .with_env_variable("UV_LINK_MODE", "copy")
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


def lint() -> dagger.Container:
    return (
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
    )


def test() -> dagger.Container:
    return (
        synced()
        # Some tests call `git ls-files`; .git isn't uploaded, so index the
        # tree (honoring .gitignore) after the package build has run.
        .with_exec(["sh", "-c", "git init -q && git add -A"])
        .with_exec(["npx", "playwright", "install", "--with-deps", "chromium"])
        .with_env_variable("XY_REQUIRE_BROWSER", "1")
        .with_exec(
            [
                "sh",
                "-c",
                'XY_CHROMIUM="$(node -e "console.log(require(\'playwright\').chromium.executablePath())")" '
                "exec .venv/bin/pytest -q",
            ]
        )
    )


CHECKS = {"rust": rust, "lint": lint, "test": test}


async def export_caches() -> None:
    # A fresh container mounting the same cache volumes, so a failed check
    # still persists what it downloaded.
    cache_dir = os.environ["CI_CACHE_DIR"]
    ctr = base().with_exec(
        [
            "sh",
            "-c",
            f"rm -rf {CACHE_OUT} && mkdir -p {CACHE_OUT} && "
            f"cp -a {CARGO_HOME}/registry {CACHE_OUT}/cargo && cp -a {UV_CACHE} {CACHE_OUT}/uv",
        ]
    )
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
