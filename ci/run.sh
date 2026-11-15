#!/usr/bin/env bash
# Run a CI check: ci/run.sh rust | lint | test | all. Needs Docker, the dagger CLI (0.21.8) and uv.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${here}/.."  # dagger scopes host-directory access to the invocation directory
exec dagger run uv run --frozen --project "${here}" python "${here}/main.py" "${@}"
