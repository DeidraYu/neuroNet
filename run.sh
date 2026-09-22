#!/usr/bin/env bash
#
# Train the network. Builds first if the executable is missing.
#
# Anything passed in is handed straight to the program:
#
#   <epochs> <miniBatchSize> <learningRate> "(layer,sizes)"
#
# With no arguments it uses the defaults in main.cpp.
#
#   ./run.sh
#   ./run.sh 1 10 3.0 "(784,30,10)"
#
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$root/build/src/neuronet" ]; then
    echo "Not built yet, doing that first."
    "$root/build.sh"
fi

exec "$root/build/src/neuronet" "$@"
