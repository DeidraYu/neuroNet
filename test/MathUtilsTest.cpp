#include "gtest/gtest.h"

#include "../src/MathUtils.hpp"
using namespace sw;

TEST(VectorViewTest, initialize)
{
    std::vector<int> std_vector_u{0, 1, 2};
    VectorView<int> u{&std_vector_u};
    EXPECT_EQ(u[0], 0);
    EXPECT_EQ(u[1], 1);
    EXPECT_EQ(u[2], 2);
}

TEST(VectorViewTest, changeOneChangesTheOther)
{
    std::vector<int> std_vector_u{0, 1, 2};
    VectorView u{&std_vector_u};

    std_vector_u[2] = 9;

    EXPECT_EQ(u[0], 0);
    EXPECT_EQ(u[1], 1);
    EXPECT_EQ(u[2], 9);
}

TEST(VectorViewTest, changeTheOtherChangesTheOne)
{
    std::vector<int> std_vector{0, 1, 2};
    VectorView vectorView{&std_vector};

    vectorView[2] = 9;

    EXPECT_EQ(std_vector[2], 9);
}

TEST(VectorViewTest, dotProduct)
{
    std::vector<int> std_vector_u{1, 2, 3};
    std::vector<int> std_vector_v{2, 3, 4};

    VectorView u{&std_vector_u};
    VectorView v{&std_vector_v};

    auto w = u * v;

    EXPECT_EQ(w, 20);
}

// Note that a point-wise multiplication yields a Vector, not a VectorView
// so, this test already tests that part of the Vector also.
TEST(VectorViewTest, pointWiseMultiplication)
{
    std::vector<int> std_vector_u{1, 2, 3};
    std::vector<int> std_vector_v{2, 3, 4};

    VectorView u{&std_vector_u};
    VectorView v{&std_vector_v};

    Vector w = u.point_mult(v);

    EXPECT_EQ(w[0], 2);
    EXPECT_EQ(w[1], 6);
    EXPECT_EQ(w[2], 12);
}

TEST(VectorTest, createWithStdVector)
{
    int dummy = 3;

    std::vector<int> std_vector_u{1, 2, 3};

    Vector u(std_vector_u);

    EXPECT_EQ(u[0], 1);
    EXPECT_EQ(u[1], 2);
    EXPECT_EQ(u[2], 3);
}

TEST(VectorTest, copyConstructor)
{
    std::vector<int> std_vector_u{1, 2, 3};

    Vector u(std_vector_u);

    Vector v{u};

    // Check correctness of the copy
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);

    // Check that all references are updated correctly
    u[0] = 4;
    u[1] = 5;
    u[2] = 6;

    v[0] = 7;
    v[1] = 8;
    v[2] = 9;

    EXPECT_EQ(u[0], 4);
    EXPECT_EQ(u[1], 5);
    EXPECT_EQ(u[2], 6);

    EXPECT_EQ(v[0], 7);
    EXPECT_EQ(v[1], 8);
    EXPECT_EQ(v[2], 9);
}

TEST(VectorTest, copyAssignmentOperator)
{
    std::vector<int> std_vector_u{1, 2, 3};

    Vector u(std_vector_u);

    Vector<int> v;
    v = u;

    // Check correctness of the copy
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);

    // Check that all references are updated correctly
    u[0] = 4;
    u[1] = 5;
    u[2] = 6;

    v[0] = 7;
    v[1] = 8;
    v[2] = 9;

    EXPECT_EQ(u[0], 4);
    EXPECT_EQ(u[1], 5);
    EXPECT_EQ(u[2], 6);

    EXPECT_EQ(v[0], 7);
    EXPECT_EQ(v[1], 8);
    EXPECT_EQ(v[2], 9);
}

TEST(VectorTest, createWithSize)
{
    Vector<double> u(3);

    u[0] = 0.0;
    u[1] = 1.0;
    u[2] = 2.0;

    EXPECT_EQ(u[0], 0.0);
    EXPECT_EQ(u[1], 1.0);
    EXPECT_EQ(u[2], 2.0);
}

TEST(VectorTest, dotProduct)
{
    Vector u{1, 2, 3};
    Vector v{2, 3, 4};

    auto w = u * v;

    EXPECT_EQ(w, 20);
}

TEST(VectorTest, pointWiseMultiplication)
{
    Vector u{1, 2, 3};
    Vector v{2, 3, 4};

    Vector w = u.point_mult(v);

    EXPECT_EQ(w[0], 2);
    EXPECT_EQ(w[1], 6);
    EXPECT_EQ(w[2], 12);
}

TEST(VectorTest, leftScalarMultiplication)
{
    Vector u{1, 2, 3};

    Vector w = 2 * u;

    EXPECT_EQ(w[0], 2);
    EXPECT_EQ(w[1], 4);
    EXPECT_EQ(w[2], 6);
}

TEST(VectorTest, rightScalarMultiplication)
{
    Vector u{1, 2, 3};

    Vector w = u * 2;

    EXPECT_EQ(w[0], 2);
    EXPECT_EQ(w[1], 4);
    EXPECT_EQ(w[2], 6);
}

TEST(VectorTest, combined)
{
    std::vector<float> std_vector_u = {1.0, 2.0, 3.0};

    Vector u(std_vector_u);
    Vector v{1, 2, 3};

    Vector w = (2 * u).point_mult(v * 2);

    EXPECT_EQ(w[0], 4.0);
    EXPECT_EQ(w[1], 16.0);
    EXPECT_EQ(w[2], 36.0);
}

TEST(VectorTest, add)
{
    Vector u{2, 4, 6};
    Vector v{1, 2, 3};

    Vector w = u + v;

    EXPECT_EQ(w[0], 3);
    EXPECT_EQ(w[1], 6);
    EXPECT_EQ(w[2], 9);
}

TEST(VectorTest, random)
{
    Vector u = Vector<uint8_t>::rand(10, 0, std::numeric_limits<uint8_t>::max());
    std::cout << u.toString() << std::endl;
}
