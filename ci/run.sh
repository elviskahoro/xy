#!/usr/bin/env bash
# Run a CI check: ci/run.sh rust | lint | test | all. Needs Docker, the dagger CLI (0.21.8) and uv.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${here}/.."  # dagger scopes host-directory access to the invocation directory
# Under CI (RWX always sets CI=true) the animated TUI just repaints progress
# frames into the log; plain output prints each step's logs once, linearly.
# DAGGER_PROGRESS overrides either way (auto, plain, tty, dots).
progress="${DAGGER_PROGRESS:-auto}"
if [[ -z "${DAGGER_PROGRESS:-}" && -n "${CI:-}" ]]; then
  progress=plain
fi
exec dagger run --progress="${progress}" uv run --frozen --quiet --project "${here}" python "${here}/main.py" "${@}"
