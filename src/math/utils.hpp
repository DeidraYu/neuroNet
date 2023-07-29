#pragma once

#include <cmath>

namespace sw
{
    namespace math
    {
        template <class T>
        concept arithmetic = std::is_arithmetic_v<T>;

        double sigmoid(double z);
        double sigmoid_prime(double z);
    } // namespace math
} // namespace sw