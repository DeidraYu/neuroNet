#include <cmath>

#include "gtest/gtest.h"

#include "../../src/math/Vector.hpp"
#include "../../src/math/Matrix.hpp"
#include "../NearComparison.hpp"

using namespace sw::math;

TEST(MatrixTest, creation)
{
    Matrix<int> A{{0, 1, 2}, {3, 4, 5}};
}

TEST(MatrixTest, MatrixVectorProduct)
{
    // Check the expression w = u + v.dot(v)

    // inputs
    Matrix<int> A{{0, 1, 2}, {3, 4, 5}};
    Vector<int> b{3, 2, 1};

    // expected results
    Vector<int> expected{4, 22};

    auto y_expression = A * b;
    EXPECT_EQ(y_expression[0], expected[0]);
    EXPECT_EQ(y_expression[1], expected[1]);

    // evaluate the whole expression tree
    Vector<int> y = y_expression;
    EXPECT_EQ(y, expected);

    // get the evaluated expression in one go
    Vector<int> z = A * b;
    EXPECT_EQ(y, expected);
}

TEST(MatrixTest, operatorPlusIs)
{
    // inputs:  A * b evaluates to [4, 22, 6]
    Matrix<int> A{{0, 1, 2}, {3, 4, 5}, {1, 1, 1}};
    Vector<int> b{3, 2, 1};

    // inputs u.dot(v) evaluates to 8.8
    Vector<int> u{0, 1, 2};
    Vector<float> v{1.1f, 2.2f, 3.3f};

    Vector<float> y(3, 1.0f); // [1.0, 1.0, 1.0]

    // expected output
    Vector<float> expected{13.8f, 31.8f, 15.8f};

    // Types are not mixed inside an operation, so each conversion is written out.
    const float s = Vector<float>(u).dot(v);
    y += Vector<float>(A * b) + s;
    EXPECT_TRUE(sw::test::vectorNear(y, expected));
}

TEST(MatrixTest, MatrixScalarAddition)
{
    // inputs:  A * b evaluates to [4, 22, 6]
    Matrix<int> A{{0, 1, 2}, {3, 4, 5}, {1, 1, 1}};

    auto W = A + 3;

    Matrix<int> expected{{3, 4, 5},
                         {6, 7, 8},
                         {4, 4, 4}};

    Matrix<int> Y = W;

    EXPECT_EQ(Y, expected);
}

TEST(MatrixTest, OuterProduct)
{
    // inputs
    Vector<int> u{1, 2, 3};
    Vector<int> v{3, 4, 5};

    Matrix<int> expected{{3, 4, 5},
                         {6, 8, 10},
                         {9, 12, 15}};

    auto W = outer(u, v);
    // auto w_0 = W[0]; // Just to see that w_0 is not evaluated yet

    Matrix<int> Y = W;
    // Vector<int> y_0 = w_0; // Just to see that y_0 is evaluated

    EXPECT_EQ(Y, expected);
}

TEST(MatrixTest, OuterProduct_and_operatorPlusIs)
{
    // inputs
    Vector<int> u{1, 2, 3};
    Vector<int> v{3, 4, 5};

    Matrix<int> Y{{3, 4, 5},
                  {6, 8, 10},
                  {9, 12, 15}};

    Matrix<int> expected{{6, 8, 10},
                         {12, 16, 20},
                         {18, 24, 30}};

    auto W = outer(u, v);
    // auto w_0 = W[0]; // Just to see that w_0 is not evaluated yet

    Y += W;
    // Vector<int> y_0 = w_0; // Just to see that y_0 is evaluated

    EXPECT_EQ(Y, expected);

    Y += outer(u, v);
}
