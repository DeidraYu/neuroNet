#include <chrono>

#include "gtest/gtest.h"

#include "../src/Matrix.hpp"

using namespace sw;

TEST(MatrixTest, initialize)
{
    Matrix<int> matrix(2, 3);

    matrix[0][2] = 9;  // top right
    matrix[1][0] = 17; // bottom left

    EXPECT_EQ(matrix[0][2], 9);
    EXPECT_EQ(matrix[1][0], 17);
}

TEST(MatrixTest, sum)
{
    Matrix<int> matrix(2, 3);

    matrix[0][2] = 9;  // top right
    matrix[1][0] = 17; // bottom left

    EXPECT_EQ(matrix.sum(), 26);
}

TEST(MatrixTest, matrixVectorMult)
{
    Matrix<int> A(2, 3);

    A[0][0] = 1;
    A[0][1] = 2;
    A[0][2] = 3;

    A[1][0] = 4;
    A[1][1] = 5;
    A[1][2] = 6;

    Vector<float> x{2.0, 3.0, 4.0};

    Vector y = A * x;

    EXPECT_EQ(y[0], 20);
    EXPECT_EQ(y[1], 47);
}

TEST(MatrixTest, copyConstructor)
{
    Matrix<int> A(2, 3);

    A[0][0] = 1;
    A[0][1] = 2;
    A[0][2] = 3;

    A[1][0] = 4;
    A[1][1] = 5;
    A[1][2] = 6;

    Matrix<int> B(A);

    // Check all entries in B are correct
    EXPECT_EQ(B[0][0], 1);
    EXPECT_EQ(B[0][1], 2);
    EXPECT_EQ(B[0][2], 3);

    EXPECT_EQ(B[1][0], 4);
    EXPECT_EQ(B[1][1], 5);
    EXPECT_EQ(B[1][2], 6);

    // Check that changes
}

TEST(MatrixTest, copyAssignment)
{
    Matrix<int> A(2, 3);

    A[0][0] = 1;
    A[0][1] = 2;
    A[0][2] = 3;

    A[1][0] = 4;
    A[1][1] = 5;
    A[1][2] = 6;

    Matrix<int> B(2, 3);
    B = A;
    Matrix<int> C(A);
    Vector<float> x{2.0, 3.0, 4.0};

    Vector y = A * x;

    // std::cout << "A: " << A.toString() << std::endl;
    // std::cout << "x: " << x.toString() << std::endl;
    // std::cout << "y: " << y.toString() << std::endl;

    EXPECT_EQ(y[0], 20);
    EXPECT_EQ(y[1], 47);
}

TEST(MatrixTest, random)
{
    Matrix A = Matrix<float>::rand(2, 3, 0.0f, 1.0f);
    std::cout << A.toString() << std::endl;
}
