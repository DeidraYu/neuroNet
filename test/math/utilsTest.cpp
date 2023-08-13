#include <cmath>

#include "gtest/gtest.h"

#include "../../src/math/Vector.hpp"
#include "../../src/math/Matrix.hpp"

using namespace sw::math;

TEST(utilsTest, random_with_random_seed)
{
    Random::seed();
    auto r = Random::gen();
    EXPECT_NE(r, 2357136044);
}

TEST(utilsTest, random_with_manual_seed)
{
    Random::seed(0);
    auto r = Random::gen();
    EXPECT_EQ(r, 2357136044);
}

TEST(utilsTest, random_with_multiple_calls)
{
    Random::seed(0);
    auto r = Random::gen();
    EXPECT_EQ(r, 2357136044);

    // Next call must generate a new value
    r = Random::gen();
    EXPECT_NE(r, 2357136044);
}

TEST(utilsTest, random_with_reseed)
{
    Random::seed(1);
    auto r = Random::gen();

    Random::seed(1);
    auto s = Random::gen();

    EXPECT_EQ(r, s);
}

TEST(utilsTest, relu)
{
    EXPECT_EQ(relu(3.14f), 3.14f);
    EXPECT_EQ(relu(0.0f), 0.0f);
    EXPECT_EQ(relu(-2.0f), 0.0f);
}

TEST(utilsTest, relu_prime)
{
    EXPECT_EQ(relu_prime(3.14f), 1.0f);
    EXPECT_EQ(relu_prime(0.0f), 0.0f);
    EXPECT_EQ(relu_prime(-2.0f), 0.0f);
}
