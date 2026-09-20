#pragma once

#include <cmath>

#include "gtest/gtest.h"

#include "../src/math/Vector.hpp"
#include "../src/math/Matrix.hpp"

/**
 * Element-wise comparison with a tolerance, for results that went through floating
 * point maths.
 *
 * Exact comparison pins down values that are allowed to differ. std::expf and friends
 * are not required to be correctly rounded, so libstdc++ and the MSVC STL return
 * results that differ by an ulp for a small fraction of inputs, and summation order
 * shifts the last bits as well. Comparing exactly turns those legal differences into
 * test failures on whichever platform the expected values were not measured on.
 */
namespace sw
{
    namespace test
    {
        constexpr double defaultTolerance = 1e-6;

        template <typename T>
        ::testing::AssertionResult vectorNear(const sw::math::Vector<T> &actual,
                                              const sw::math::Vector<T> &expected,
                                              double tolerance = defaultTolerance)
        {
            if (actual.size() != expected.size())
            {
                return ::testing::AssertionFailure()
                       << "sizes differ: " << actual.size() << " vs " << expected.size();
            }

            for (size_t i = 0; i < actual.size(); ++i)
            {
                const double difference = std::abs(static_cast<double>(actual[i]) -
                                                   static_cast<double>(expected[i]));
                if (difference > tolerance)
                {
                    return ::testing::AssertionFailure()
                           << "element " << i << " differs: " << actual[i] << " vs " << expected[i]
                           << " (|difference| = " << difference << " > " << tolerance << ")";
                }
            }
            return ::testing::AssertionSuccess();
        }

        template <typename T>
        ::testing::AssertionResult matrixNear(const sw::math::Matrix<T> &actual,
                                              const sw::math::Matrix<T> &expected,
                                              double tolerance = defaultTolerance)
        {
            if (actual.getNumRows() != expected.getNumRows() ||
                actual.getNumCols() != expected.getNumCols())
            {
                return ::testing::AssertionFailure()
                       << "shapes differ: " << actual.getNumRows() << "x" << actual.getNumCols()
                       << " vs " << expected.getNumRows() << "x" << expected.getNumCols();
            }

            for (size_t r = 0; r < actual.getNumRows(); ++r)
            {
                const ::testing::AssertionResult rowResult =
                    vectorNear(actual[r], expected[r], tolerance);
                if (!rowResult)
                {
                    return ::testing::AssertionFailure() << "row " << r << ": " << rowResult.message();
                }
            }
            return ::testing::AssertionSuccess();
        }

    } // namespace test
} // namespace sw
