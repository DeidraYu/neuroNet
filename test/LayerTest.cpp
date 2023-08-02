#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "../src/Layer.hpp"
#include "../src/Network.hpp"

#include "../src/math/Vector.hpp"

using namespace sw;

class LayerTest : public ::testing::Test
{
protected:
    // Store the original value of 'verbosity' in a member variable
    int originalVerbosity;

    void SetUp() override
    {
        // Set up for each test: Overwrite 'verbosity' to 0 and store the original value
        originalVerbosity = sw::verbosity;
        sw::verbosity = 0;
    }

    void TearDown() override
    {
        // Tear down after each test: Restore 'verbosity' to its original value
        sw::verbosity = originalVerbosity;
    }
};

TEST_F(LayerTest, initialize)
{
    Layer layer = Layer(2, 2);
}

TEST_F(LayerTest, feedForward)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer = Layer(2, 2);

    Vector<float> x{0.3f, 0.9f};

    layer.feedForward(x);

    // Expect that the z_j are the average of 0.3 and 0.9.
    // So, that the output vector consists of sigmoid([0.6, 0.6]) = [0.646, 0.646]
    EXPECT_NEAR(layer.getY()[0], 0.645656, 0.000001);
    EXPECT_NEAR(layer.getY()[1], 0.645656, 0.000001);
}

TEST_F(LayerTest, backProp)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer = Layer(2, 2);

    // Set the input vector
    Vector<float> x{0.3f, 0.9f};

    // Perform the feed forward
    layer.feedForward(x);

    // Compute the loss
    Vector<float> v = layer.getY() - x; // u = dC / dx

    // Compute the back propagation
    layer.backProp(x, v);

    // The expected values
    Vector<float> nablaC_b_expected = v * sigmoid_prime(0.6);
    Matrix<float> nablaC_W_expected = outer(nablaC_b_expected, x);
    Vector u_expected = Matrix<float>(2, 2, 0.5f).transposeMult(nablaC_b_expected);

    // Check
    EXPECT_EQ(layer.getNablaC_b(), nablaC_b_expected);
    EXPECT_EQ(layer.getNablaC_W(), nablaC_W_expected);
    EXPECT_EQ(layer.getU(), u_expected);
}

TEST_F(LayerTest, update)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer = Layer(2, 2);

    Vector x{0.3f, 0.9f};

    Vector<float> v;

    for (int i = 0; i < 100; i++)
    {

        layer.feedForward(x);
        v = layer.getY() - x; // u = dC / dx
        layer.backProp(x, v);
        layer.update(1.0f);
    }

    float cost = 0.5f * static_cast<float>(v.dot(v));

    EXPECT_LE(cost, 1e-3);
}

TEST_F(LayerTest, update2)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer = Layer(2, 2);

    Vector x1{0.3f, 0.9f};
    Vector x2{0.8f, 0.6f};

    Vector<float> v;
    float cost1, cost2;

    for (int i = 0; i < 2000; i++)
    {
        {
            layer.feedForward(x1);
            v = layer.getY() - x1; // u = dC / dx
            cost1 = 0.5f * static_cast<float>(v.dot(v));
            layer.backProp(x1, v);
            layer.update(1.0f);
        }

        {
            layer.feedForward(x2);
            v = layer.getY() - x2; // u = dC / dx
            cost2 = 0.5f * static_cast<float>(v.dot(v));
            layer.backProp(x2, v);
            layer.update(1.0f);
        }
    }

    EXPECT_LE(cost1, 1e-6);
    EXPECT_LE(cost2, 1e-6);
}
