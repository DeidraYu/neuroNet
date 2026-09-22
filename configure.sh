#!/usr/bin/env bash
#
# Configure the CMake build.
#
# Only needs to run once, or again after CMakeLists.txt changes.
#
#   ./configure.sh              # tuned for this machine
#   ./configure.sh --no-tuning  # portable binaries
#
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mnist="$(dirname "$root")/mnist"
if [ ! -d "$mnist" ]; then
    echo "The MNIST loader is missing. It has to sit next to this repository:"
    echo "    cd $(dirname "$root")"
    echo "    git clone https://github.com/wichtounet/mnist.git"
    exit 1
fi

extra=()
if [ "${1:-}" = "--no-tuning" ]; then
    extra+=(-DNEURONET_TUNE_FOR_THIS_MACHINE=OFF)
fi

compiler=clang++
command -v clang++ >/dev/null 2>&1 || compiler=g++

echo "Configuring with $compiler ..."
cmake -S "$root" -B "$root/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$compiler" \
      "${extra[@]}"

echo "Configured. Next: ./build.sh"
