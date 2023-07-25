#include "gtest/gtest.h"

#include "../../src/math/Vector.hpp"

using namespace math;

TEST(VectorViewTest, initialize)
{
    std::vector<int> std_vector_u{0, 1, 2};
    Vector<int> u(std_vector_u);
    EXPECT_EQ(u[0], 0);
    EXPECT_EQ(u[1], 1);
    EXPECT_EQ(u[2], 2);
}

TEST(VectorViewTest, VectorExpression)
{
    std::vector<int> std_vector_u{0, 1, 2};
    Vector<int> u(std_vector_u);

    VectorExpression<Vector<int>> e = *(static_cast<VectorExpression<Vector<int>> *>(&u));

    EXPECT_EQ(e[0], 0);
    EXPECT_EQ(e[1], 1);
    EXPECT_EQ(e[2], 2);
}