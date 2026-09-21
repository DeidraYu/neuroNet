#include "utils.hpp"

#include <cassert>
#include <cmath>
#include <random>
#include <typeindex>
#include <vector>
#include <variant>
#include <stdexcept>

struct TypeInfo
{
    std::type_index index;
    size_t size;
};

namespace sw
{
    namespace math
    {
        // NOINLINE template <arithmetic T>
        // T sigmoid(T z)
        // {
        //     return T{1} / (T{1} + exp(-z));
        // }

        // NOINLINE double sigmoid_prime(double z)
        // {
        //     double s = 1.0 / (1.0 + exp(-z));
        //     return (1.0 - s) * s;
        // }

        void Random::seed()
        {
            std::random_device rd;
            seed(rd());
        }

        void Random::seed(uint32_t seed)
        {
            gen = std::mt19937(seed);
        }

        uint32_t Random::below(uint32_t bound)
        {
            if (bound <= 1)
            {
                return 0;
            }

            // The engine produces every value in [0, 2^32). Taking that modulo `bound`
            // would favour the first 2^32 % bound values, so we discard the values that
            // do not fit a whole number of times and draw again.
            const uint64_t range = 1ULL << 32;
            const uint64_t usable = range - (range % bound);

            uint64_t draw;
            do
            {
                draw = static_cast<uint32_t>(gen());
            } while (draw >= usable);

            return static_cast<uint32_t>(draw % bound);
        }

        std::mt19937 Random::gen = std::mt19937(0);

    } // namespace math
} // namespace sw