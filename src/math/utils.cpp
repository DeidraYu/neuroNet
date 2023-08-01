#include "utils.hpp"

#include <cmath>
#include <random>

namespace sw
{
    namespace math
    {
        NOINLINE double sigmoid(double z)
        {
            return 1.0 / (1.0 + exp(-z));
        }

        NOINLINE double sigmoid_prime(double z)
        {
            double s = 1.0 / (1.0 + exp(-z));
            return (1.0 - s) * s;
        }

        void Random::seed()
        {
            std::random_device rd;
            seed(rd());
        }

        void Random::seed(uint32_t seed)
        {
            gen = std::mt19937(seed);
        }

        std::mt19937 Random::gen = std::mt19937(0);

    } // namespace math
} // namespace sw