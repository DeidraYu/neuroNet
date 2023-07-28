#pragma once

#include <cmath>

namespace math
{
    template <class T>
    concept arithmetic = std::is_arithmetic_v<T>;

    template <arithmetic T>
    T sigmoid(T z)
    {
        return T{1} / (T{1} + static_cast<float>(exp(-z)));
    }

    double sigmoid_prime(double z)
    {
        double s = 1.0 / (1.0 + exp(-z));
        return (1.0f - s) * s;
    }

} // namespace math
