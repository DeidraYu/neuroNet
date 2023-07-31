#pragma once

#include <algorithm> // for the std::unordered_map
#include <chrono>    // for the profiling classes
#include <cmath>
#include <string>

#define NOINLINE
#ifndef NOINLINE
#ifdef _MSC_VER // Check if compiling with Microsoft Visual C++
#define NOINLINE __declspec(noinline)
#else // Assume GCC or Clang for Linux and other platforms
#define NOINLINE __attribute__((noinline))
#endif
#endif

namespace sw
{
    namespace math
    {
        template <class T>
        concept arithmetic = std::is_arithmetic_v<T>;

        NOINLINE double sigmoid(double z);
        NOINLINE double sigmoid_prime(double z);
    } // namespace math
} // namespace sw
