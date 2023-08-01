#include <cmath>

#include "gtest/gtest.h"

#include "../../src/math/Vector.hpp"
#include "../../src/math/Matrix.hpp"

using namespace sw::math;

TEST(VectorTest, initializeWithStdVector)
{
    std::vector<int> std_vector_u{0, 1, 2};
    Vector<int> u(std_vector_u);
    EXPECT_EQ(u[0], 0);
    EXPECT_EQ(u[1], 1);
    EXPECT_EQ(u[2], 2);
}

TEST(VectorTest, initializeWithInitializerList)
{
    Vector<int> u{0, 1, 2};
    EXPECT_EQ(u[0], 0);
    EXPECT_EQ(u[1], 1);
    EXPECT_EQ(u[2], 2);
}

TEST(VectorTest, operatorIsIs)
{
    Vector<int> u{0, 1, 2};
    EXPECT_EQ(u, Vector<int>({0, 1, 2}));
}

TEST(VectorTest, VectorVectorAddition)
{
    Vector<int> u{0, 1, 2};
    Vector<int> v{3, 4, 5};

    VectorVectorAddition<Vector<int>, Vector<int>> vectorAddition(u, v);
}

TEST(VectorTest, operatorPlus_00)
{
    Vector<int> u{0, 1, 2};
    Vector<int> v{3, 4, 5};

    auto w = u + v;

    EXPECT_EQ(Vector<int>(w), Vector<int>({3, 5, 7}));
}

TEST(VectorTest, operatorPlus_01)
{
    Vector<int> u{0, 1, 2};
    Vector<int> v{3, 4, 5};

    Vector<int> w = u + v;

    EXPECT_EQ(w, Vector<int>({3, 5, 7}));
}

TEST(VectorTest, mixed_types)
{
    Vector<int> u{0, 1, 2};
    Vector<float> v{3.f, 4.f, 5.f};

    auto w = u + v;

    EXPECT_EQ(w, Vector<float>({3.f, 5.f, 7.f}));
}

TEST(VectorTest, mixed_types_with_eval)
{
    Vector<int> u{0, 1, 2};
    Vector<float> v{3.1f, 4.f, 5.f};

    Vector<float> w = u + v;

    EXPECT_EQ(w, Vector<float>({3.1f, 5.f, 7.f}));
}

TEST(VectorTest, VectorScalarAddition_construction)
{
    Vector<int> u{0, 1, 2};
    float s = 1.1f;

    VectorScalarAddition<Vector<int>, float> vectorScalarAddition(u, s);
}

TEST(VectorTest, VectorScalarAddition)
{
    // inputs
    Vector<int> u{0, 1, 2};
    float s = 1.1f;

    // expected output
    Vector<float> expected{1.1f, 2.1f, 3.1f};

    VectorScalarAddition<Vector<int>, float> y = u + s;
    EXPECT_EQ(Vector<float>(y), expected);

    auto w = u + s;
    EXPECT_EQ(Vector<float>(w), expected);

    Vector<float> z = u + s;
    EXPECT_EQ(z, expected);
}

TEST(VectorTest, VectorVectorMultiplication_construction)
{
    Vector<int> u{0, 1, 2};
    Vector<float> v{3.1f, 4.f, 5.f};

    VectorVectorMultiplication<Vector<int>, Vector<float>>(u, v);
}

TEST(VectorTest, VectorVectorMultiplication)
{
    // inputs
    Vector<int> u{0, 1, 2};
    Vector<float> v{1.1f, 2.2f, 3.3f};

    // expected output
    Vector<float> expected{0.f, 2.2f, 6.6f};

    // explicit type of expression
    VectorVectorMultiplication<Vector<int>, Vector<float>> y = u * v;
    EXPECT_EQ(Vector<float>(y), expected);

    // auto type of expression
    auto w = u * v;
    EXPECT_EQ(Vector<float>(w), expected);

    // evaluated expression
    Vector<float> z = u * v;
    EXPECT_EQ(z, expected);
}

TEST(VectorTest, VectorScalarMultiplication_construction)
{
    Vector<int> u{0, 1, 2};
    float s = 1.1f;

    VectorScalarMultiplication<Vector<int>, float> vectorScalarMultiplication(u, s);
}

TEST(VectorTest, VectorScalarMultiplication)
{
    Vector<int> u{0, 1, 2};
    float s = 1.1f;

    // expected output
    Vector<float> expected{0.f, 1.1f, 2.2f};

    // explicit type of expression
    VectorScalarMultiplication<Vector<int>, float> y = u * s;
    EXPECT_EQ(Vector<float>(y), expected);

    // auto type of expression
    auto w = u * s;
    EXPECT_EQ(Vector<float>(w), expected);

    // evaluated expression
    Vector<float> z = u * s;
    EXPECT_EQ(z, expected);
}

TEST(VectorTest, VectorDotProduct)
{
    // Check the expression w = u + v.dot(v)

    // inputs
    Vector<int> u{0, 1, 2};
    Vector<float> v{1.1f, 2.2f, 3.3f};

    // expected output
    Vector<float> expected{8.8f, 9.8f, 10.8f};

    Vector<float> y = u + u.dot(v);
    EXPECT_EQ(y, expected);
}

TEST(VectorTest, composition)
{
    Vector<int> u{0, 1, 2};
    Vector<int> v{3, 4, 5};
    auto x = u + v; // [3, 5, 7]
    auto y = x * x; // [9, 25, 49]

    Vector<int> z = y; // In total 6 scalar additions are done because of (u + v) * (u + v)

    EXPECT_EQ(z, Vector<int>({9, 25, 49}));
}

TEST(VectorTest, operatorPlusIs_with_vector)
{
    // Check the expression w = u + v.dot(v)

    // inputs
    Vector<int> u{0, 1, 2};
    Vector<float> v{1.1f, 2.2f, 3.3f};

    Vector<float> y(3, 1.0f); // [1.0, 1.0, 1.0]

    // expected output
    Vector<float> expected{9.8f, 10.8f, 11.8f};

    y += u + u.dot(v);
    EXPECT_EQ(y, expected);
}

TEST(VectorTest, operatorPlusIs_with_scalar)
{
    // inputs
    Vector<float> v{1.1f, 2.2f, 3.3f};

    v += 2.0f;

    Vector<float> expected{3.1f, 4.2f, 5.3f};

    EXPECT_EQ(v, expected);
}

TEST(VectorTest, operatorTimesIs_with_vector)
{
    // inputs
    Vector<float> u{2.0f, 2.0f, 2.0f};
    Vector<float> v{1.1f, 2.2f, 3.3f};

    u *= v;

    Vector<float> expected{2.2f, 4.4f, 6.6f};

    EXPECT_EQ(u, expected);
}

TEST(VectorTest, operatorTimesIs_with_scalar)
{
    // inputs
    Vector<float> v{1.1f, 2.2f, 3.3f};

    v *= 2.0f;

    Vector<float> expected{2.2f, 4.4f, 6.6f};

    EXPECT_EQ(v, expected);
}

TEST(VectorTest, sigmoid)
{
    // inputs
    // Force computation to be done on a double vector.
    Vector<double> v{0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

    auto sigmoid_v_expression = sigmoid(v);

    // Now create the float version for comparison.
    Vector<float> sigmoid_v = sigmoid_v_expression;

    Vector<float> expected{0.5f, 0.5498339973124953f, 0.5986876601124858f, 0.6456563062258437f, 0.6899744811276725f, 0.731058578630074f};
    // Vector<float> expected{0.25f, 0.2475165727118582f, 0.24026074574152248f, 0.22878424045664325f, 0.2139096965202716f, 0.19661193324144993f};

    EXPECT_EQ(sigmoid_v, expected);
}

TEST(VectorTest, sigmoid_prime)
{
    // inputs
    // Force computation to be done on a double vector.
    Vector<double> v{0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

    auto sigmoid_prime_v_expression = sigmoid_prime(v);

    // Now create the float version for comparison.
    Vector<float> sigmoid_prime = sigmoid_prime_v_expression;

    // Vector<float> expected{0.5f, 0.5498339973124953f, 0.5986876601124858f, 0.6456563062258437f, 0.6899744811276725f, 0.731058578630074f};
    Vector<float> expected{0.25f, 0.2475165727118582f, 0.24026074574152248f, 0.22878424045664325f, 0.2139096965202716f, 0.19661193324144993f};

    EXPECT_EQ(sigmoid_prime, expected);
}

TEST(VectorTest, VectorFunction_sin)
{
    // inputs
    Vector<float> v{0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

    auto sin_v_expression = sin(v);

    // Now create the float version for comparison.
    Vector<float> sin_v = sin_v_expression;

    Vector<float> expected{0.f, 0.19866933f, 0.38941834f, 0.56464247f, 0.71735609f, 0.84147098f};

    EXPECT_EQ(sin_v, expected);
}

TEST(VectorTest, VectorFunction_cos)
{
    // inputs
    Vector<float> v{0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

    auto cos_v_expression = cos(v);

    // Evaulate the expression
    Vector<float> cos_v = cos_v_expression;

    Vector<float> expected{1.f, cosf(0.2f), cosf(0.4f), cosf(0.6f), cosf(0.8f), cosf(1.0f)};

    EXPECT_EQ(cos_v, expected);
}

#if 0 // for documentation
TEST(VectorTest, CodeForMdFile)
{
    // create and initialize two vectors
    Vector<int> u{1, 2, 3};
    Vector<int> v{4, 5, 6};

    // add them together
    auto w = u + v;
    Vector<int> x = u + v;
    Vector<int> y{u + v};

    // print the results
    std::cout << "w: " << x << std::endl;
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << x << std::endl;
}
#endif

TEST(VectorTest, random)
{
    Vector u = Vector<float>::rand(10, 0, std::numeric_limits<float>::max());

    bool isRandom = u[0] != u[2];

    EXPECT_EQ(isRandom, true);
}

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

    y += A * b + u.dot(v);
    EXPECT_EQ(y, expected);
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
