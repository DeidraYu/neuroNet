#include <cmath>

#include "gtest/gtest.h"

#include "../../src/math/Vector.hpp"
#include "../../src/math/Matrix.hpp"
#include "../../src/FileIO.hpp"

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

TEST(VectorTest, VectorToBinary)
{
    // inputs
    Vector<uint16_t> v{1, 2, 3, 4, 5, 8};

    std::vector<uint8_t> b = v.getBinaryString();

    size_t typeSize = sizeof(TypeInfo);
    size_t numElementSize = sizeof(size_t);
    size_t dataSize = sizeof(uint16_t) * v.size();

    EXPECT_EQ(b.size(), typeSize + numElementSize + dataSize);
}

TEST(VectorTest, BinaryToVector)
{
    // inputs
    Vector<uint16_t> v{1, 2, 3, 4, 5, 8};

    std::vector<uint8_t> b = v.getBinaryString();

    Vector<uint16_t> vecFromBin = parseBinaryData<uint16_t>(b);

    EXPECT_EQ(vecFromBin, v);
}

TEST(VectorTest, BinaryToVectorMultipleTypes)
{
    {
        Vector<uint8_t> v{3, 1, 4, 1, 5};
        std::vector<uint8_t> b = v.getBinaryString();
        Vector<uint8_t> vecFromBin = parseBinaryData<uint8_t>(b);
        EXPECT_EQ(vecFromBin, v);
    }
    {
        Vector<uint16_t> v{3, 1, 4, 1, 5};
        std::vector<uint8_t> b = v.getBinaryString();
        Vector<uint16_t> vecFromBin = parseBinaryData<uint16_t>(b);
        EXPECT_EQ(vecFromBin, v);
    }
    {
        Vector<size_t> v{3, 1, 4, 1, 5};
        std::vector<uint8_t> b = v.getBinaryString();
        Vector<size_t> vecFromBin = parseBinaryData<size_t>(b);
        EXPECT_EQ(vecFromBin, v);
    }
    {
        Vector<float> v{3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
        std::vector<uint8_t> b = v.getBinaryString();
        Vector<float> vecFromBin = parseBinaryData<float>(b);
        EXPECT_EQ(vecFromBin, v);
    }
    {
        Vector<double> v{3.0, 1.0, 4.0, 1.0, 5.0};
        std::vector<uint8_t> b = v.getBinaryString();
        Vector<double> vecFromBin = parseBinaryData<double>(b);
        EXPECT_EQ(vecFromBin, v);
    }
    /*{
        Matrix<double> m{{3.0, 1.0, 4.0, 1.0, 5.0}, {2.0, 7.0, 1.0, 2.0, 8.0}};
        std::vector<uint8_t> b = m.getBinaryString();
        Vector<double> vecFromBin = parseBinaryData<double>(b);
        EXPECT_EQ(vecFromBin, m);
    }*/
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
