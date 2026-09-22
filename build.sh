#!/usr/bin/env bash
#
# Build the project. Configures first if that has not been done yet.
#
#   ./build.sh
#
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$root/build/CMakeCache.txt" ]; then
    echo "Not configured yet, doing that first."
    "$root/configure.sh"
fi

echo "Building ..."
cmake --build "$root/build"

echo "Built. Next: ./run.sh or ./test.sh"
