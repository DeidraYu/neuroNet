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

TEST(MatrixText, operatorEquals_false)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{1, 2, 3},
                  {4, 5, 0}};

    bool isEqual = (A == B);

    EXPECT_EQ(isEqual, false);
}

TEST(MatrixText, operatorEquals_true)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{1, 2, 3},
                  {4, 5, 6}};

    bool isEqual = (A == B);

    EXPECT_EQ(isEqual, true);
}

TEST(MatrixText, operatorNotEquals_true)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{1, 2, 3},
                  {4, 5, 0}};

    bool isNotEqual = (A != B);

    EXPECT_EQ(isNotEqual, true);
}

TEST(MatrixText, operatorNotEquals_false)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{1, 2, 3},
                  {4, 5, 6}};

    bool isNotEqual = (A != B);

    EXPECT_EQ(isNotEqual, false);
}

TEST(MatrixTest, initializerList)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    // Check all entries in A are correct

    EXPECT_EQ(A, Matrix<int>({{1, 2, 3},
                              {4, 5, 6}}));
}

TEST(MatrixTest, createWithSizeAndInitialValue)
{
    Matrix A(2, 3, 1.0f);

    // Check all entries in A are correct
    EXPECT_EQ(A, Matrix<float>({{1.0f, 1.0f, 1.0f},
                                {1.0f, 1.0f, 1.0f}}));
}

TEST(MatrixTest, copyConstructor)
{
    Matrix<int> A{{1, 2, 3}, {4, 5, 6}};

    Matrix<int> B(A);

    // Check all entries in B are correct
    EXPECT_EQ(B, Matrix<int>({{1, 2, 3},
                              {4, 5, 6}}));

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
    EXPECT_EQ(B, Matrix<int>({{1, 2, 3},
                              {4, 5, 6}}));

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

    EXPECT_EQ(C, Matrix<int>({{3, 6, 4},
                              {6, 10, 7}}));
}

TEST(MatrixTest, subtract)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{2, 4, 1},
                  {2, 5, 1}};

    Matrix C = A - B;

    EXPECT_EQ(C, Matrix<int>({{-1, -2, 2},
                              {2, 0, 5}}));
}

TEST(MatrixTest, matrixVectorMult)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Vector<float> x{2.0, 3.0, 4.0};

    Vector y = A * x;

    EXPECT_EQ(y, Vector<float>({20, 47}));
}

TEST(MatrixTest, matrixTransposeMult)
{
    Matrix<int> A{{1, 4},
                  {2, 5},
                  {3, 6}};

    Vector<float> x{2.0, 3.0, 4.0};

    Vector y = A.transposeMult(x);

    EXPECT_EQ(y, Vector<float>({20, 47}));
}

TEST(MatrixTest, matrixTransposeMultVectorView)
{
    Matrix<int> A{{1, 4},
                  {2, 5},
                  {3, 6}};

    std::vector<float> std_vector_x{2.0, 3.0, 4.0};
    Vector<float> x(std_vector_x);

    Vector y = A.transposeMult(x);

    EXPECT_EQ(y, Vector<float>({20, 47}));
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

    EXPECT_EQ(A, Matrix<float>({{9, 9, 9},
                                {9, 9, 9}}));
}

TEST(MatrixTest, rightMultiplyByScalar)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    A = A * 2;

    EXPECT_EQ(A, Matrix<int>({{2, 4, 6},
                              {8, 10, 12}}));
}

TEST(MatrixTest, operatorPlusIsScalar)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    A += 2;

    EXPECT_EQ(A, Matrix<int>({{3, 4, 5},
                              {6, 7, 8}}));
}

TEST(MatrixTest, operatorMinusIsScalar)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    A -= 1;

    EXPECT_EQ(A, Matrix<int>({{0, 1, 2},
                              {3, 4, 5}}));
}

TEST(MatrixTest, operatorMultiplyIsScalar)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    A *= 2;

    EXPECT_EQ(A, Matrix<int>({{2, 4, 6},
                              {8, 10, 12}}));
}

TEST(MatrixTest, operatorPlusIs)
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};

    Matrix<int> B{{2, 4, 1},
                  {2, 5, 1}};

    A += B;

    EXPECT_EQ(A, Matrix<int>({{3, 6, 4},
                              {6, 10, 7}}));
}

TEST(MatrixTest, plusIsOuter)
{
    Matrix<int> A(3, 2, 1);

    Vector u{2, 5, 8};
    Vector v{1, 2};

    A.plusIsOuter(u, v);

    EXPECT_EQ(A, Matrix<int>({{3, 5},
                              {6, 11},
                              {9, 17}}));
}