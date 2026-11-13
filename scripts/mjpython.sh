#!/usr/bin/env bash
# Launch a script under MuJoCo's mjpython (needed for the native viewer on macOS).
# uv's standalone Python keeps libpython in its own lib dir, which mjpython cannot find
# through @rpath, so we expose it via DYLD_LIBRARY_PATH.
#   ./scripts/mjpython.sh scripts/run_demo.py --gui --perception oracle
set -euo pipefail
cd "$(dirname "$0")/.."
LIBDIR="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export DYLD_LIBRARY_PATH="${LIBDIR}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec uv run mjpython "$@"
