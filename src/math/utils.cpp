#include "utils.hpp"

#include <cmath>

namespace sw
{
    namespace math
    {
        double sigmoid(double z)
        {
            return 1.0 / (1.0 + exp(-z));
        }

        double sigmoid_prime(double z)
        {
            double s = 1.0 / (1.0 + exp(-z));
            return (1.0 - s) * s;
        }

    } // namespace math
} // namespace sw