#pragma once

#include <cmath>
#include <random>

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

        // NOINLINE template <arithmetic T>
        // T sigmoid(T z);

        NOINLINE template <arithmetic T>
        T sigmoid(T z)
        {
            // bool isFloat = std::is_same_v<T, float>;
            // bool isDouble = std::is_same_v<T, double>;

            if constexpr (std::is_same_v<T, float>)
                return T{1} / (T{1} + expf(-z));
            else
                return T{1} / (T{1} + exp(-z));
        }

        // NOINLINE double sigmoid_prime(double z);

        NOINLINE template <arithmetic T>
        T sigmoid_prime(T z)
        {
            T s = sigmoid(z);
            return (T{1} - s) * s;
        }

        class Random
        {
        public:
            /** @brief Set a random seed for the random number generator.*/
            static void seed();

            /** @brief Set the seed of the random number generator.*/
            static void seed(uint32_t seed);

            /**
             * @brief A uniformly distributed integer in [0, bound).
             *
             * std::uniform_int_distribution and std::shuffle are both left implementation
             * defined by the standard, so they hand out different numbers on libstdc++ and
             * on the MSVC STL. We draw from the engine, which is specified exactly, and
             * reject the tail that would not divide evenly, which keeps the result both
             * uniform and identical on every platform.
             */
            static uint32_t below(uint32_t bound);

            /** @brief The random number generator which is seeded by 0 by default.*/
            static std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
        };

    } // namespace math
} // namespace sw
