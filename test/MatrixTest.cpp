#include <chrono>

#include "gtest/gtest.h"

#include "../src/Matrix.hpp"

using namespace sw;

TEST(MatrixTest, createWithSize)
{
    Matrix<int> matrix(2, 3);

    matrix[0][2] = 9;  // top right
    matrix[1][0] = 17; // bottom left

    EXPECT_EQ(matrix[0][2], 9);
    EXPECT_EQ(matrix[1][0], 17);
}

TEST(MatrixTest, initializerList)
{
    Matrix<int> A{{1, 2, 3}, {4, 5, 6}};

    // Check all entries in A are correct
    EXPECT_EQ(A[0][0], 1);
    EXPECT_EQ(A[0][1], 2);
    EXPECT_EQ(A[0][2], 3);

    EXPECT_EQ(A[1][0], 4);
    EXPECT_EQ(A[1][1], 5);
    EXPECT_EQ(A[1][2], 6);
}

TEST(MatrixTest, createWithSizeAndInitialValue)
{
    Matrix A(2, 3, 1.0f);

    // Check all entries in A are correct
    EXPECT_EQ(A[0][0], 1.0f);
    EXPECT_EQ(A[0][1], 1.0f);
    EXPECT_EQ(A[0][2], 1.0f);

    EXPECT_EQ(A[1][0], 1.0f);
    EXPECT_EQ(A[1][1], 1.0f);
    EXPECT_EQ(A[1][2], 1.0f);
}

TEST(MatrixTest, copyConstructor)
{
    Matrix<int> A{{1, 2, 3}, {4, 5, 6}};

    Matrix<int> B(A);

    // Check all entries in B are correct
    EXPECT_EQ(B[0][0], 1);
    EXPECT_EQ(B[0][1], 2);
    EXPECT_EQ(B[0][2], 3);

    EXPECT_EQ(B[1][0], 4);
    EXPECT_EQ(B[1][1], 5);
    EXPECT_EQ(B[1][2], 6);

    // Check that changes in A are not visible in B
    A[0][0] = 9;
    EXPECT_EQ(B[0][0], 1);
}

TEST(MatrixTest, copyAssignment)
{
    Matrix<int> A{{1, 2, 3}, {4, 5, 6}};

    Matrix<int> B(2, 3);
    B = A;

    // Check all entries in B are correct
    EXPECT_EQ(B[0][0], 1);
    EXPECT_EQ(B[0][1], 2);
    EXPECT_EQ(B[0][2], 3);

    EXPECT_EQ(B[1][0], 4);
    EXPECT_EQ(B[1][1], 5);
    EXPECT_EQ(B[1][2], 6);

    // Check that changes in A are not visible in B
    A[0][0] = 9;
    EXPECT_EQ(B[0][0], 1);
}

TEST(MatrixTest, sum)
{
    Matrix<int> A{{1, 2, 3}, {4, 5, 6}};

    EXPECT_EQ(A.sum(), 21);
}

TEST(MatrixTest, add)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{2, 4, 1},
                  {2, 5, 1}};

    Matrix C = A + B;

    EXPECT_EQ(C[0][0], 3);
    EXPECT_EQ(C[0][1], 6);
    EXPECT_EQ(C[0][2], 4);

    EXPECT_EQ(C[1][0], 6);
    EXPECT_EQ(C[1][1], 10);
    EXPECT_EQ(C[1][2], 7);
}

TEST(MatrixTest, subtract)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{2, 4, 1},
                  {2, 5, 1}};

    Matrix C = A - B;

    EXPECT_EQ(C[0][0], -1);
    EXPECT_EQ(C[0][1], -2);
    EXPECT_EQ(C[0][2], 2);

    EXPECT_EQ(C[1][0], 2);
    EXPECT_EQ(C[1][1], 0);
    EXPECT_EQ(C[1][2], 5);
}

TEST(MatrixTest, matrixVectorMult)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Vector<float> x{2.0, 3.0, 4.0};

    Vector y = A * x;

    EXPECT_EQ(y[0], 20);
    EXPECT_EQ(y[1], 47);
}

TEST(MatrixTest, matrixTransposeMult)
{
    Matrix<int> A{{1, 4},
                  {2, 5},
                  {3, 6}};

    Vector<float> x{2.0, 3.0, 4.0};

    Vector y = A.transposeMult(x);

    EXPECT_EQ(y[0], 20);
    EXPECT_EQ(y[1], 47);
}

TEST(MatrixTest, matrixTransposeMultVectorView)
{
    Matrix<int> A{{1, 4},
                  {2, 5},
                  {3, 6}};

    std::vector<float> std_vector_x{2.0, 3.0, 4.0};
    Vector<float> x(std_vector_x);

    Vector y = A.transposeMult(x);

    EXPECT_EQ(y[0], 20);
    EXPECT_EQ(y[1], 47);
}

TEST(MatrixTest, random)
{
    Matrix A = Matrix<float>::rand(2, 3, 0.0f, 1.0f);

    bool isRandom = A[0][0] != A[1][2];

    EXPECT_EQ(isRandom, true);
}

TEST(MatrixTest, fill)
{
    Matrix A = Matrix<float>::rand(2, 3, 0.0f, 1.0f);

    A.fill(9);

    EXPECT_EQ(A[0][0], 9);
    EXPECT_EQ(A[0][1], 9);
    EXPECT_EQ(A[0][2], 9);
}
