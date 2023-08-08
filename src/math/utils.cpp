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

        uint32_t Random::seed()
        {
            std::random_device rd;
            uint32_t rdSeed = rd();
            seed(rdSeed);
            return rdSeed;
        }

        uint32_t Random::seed(uint32_t seed)
        {
            gen = std::mt19937(seed);
            return seed;
        }

        std::mt19937 Random::gen = std::mt19937(0);

    } // namespace math
} // namespace sw