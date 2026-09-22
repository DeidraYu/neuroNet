#!/usr/bin/env bash
#
# Run the test suite. Builds first if that has not been done yet.
# Failing tests print their output.
#
#   ./test.sh
#   ./test.sh network      # only tests whose name matches
#
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$root/build/CMakeCache.txt" ]; then
    echo "Not built yet, doing that first."
    "$root/build.sh"
fi

args=(--test-dir "$root/build" --output-on-failure)
if [ $# -gt 0 ]; then
    args+=(-R "$1")
fi

if ctest "${args[@]}"; then
    echo "All tests passed."
else
    echo "Tests failed."
    exit 1
fi
