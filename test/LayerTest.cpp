#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "../src/Layer.hpp"
#include "../src/Network.hpp"

using namespace sw;

TEST(LayerTest, initialize)
{
    Layer layer = Layer(2, 2);
}

TEST(LayerTest, feedForward)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer = Layer(2, 2);

    Vector x{0.3f, 0.9f};

    layer.feedForward(x);

    // Expect that the z_j are the average of 0.3 and 0.9.
    // So, that the output vector consists of sigmoid([0.6, 0.6]) = [0.646, 0.646]
    EXPECT_NEAR(layer.getY()[0], 0.645656, 0.000001);
    EXPECT_NEAR(layer.getY()[1], 0.645656, 0.000001);
}

TEST(LayerTest, backProp)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer = Layer(2, 2);

    Vector x{0.3f, 0.9f};

    layer.feedForward(x);

    Vector v = layer.getY() - x; // u = dC / dx

    layer.backProp(x, v);

    // const Matrix M = layer.getNablaC_W();
    // const std::string s = M.toString();

    std::cout << "nablaC_W: " << layer.getNablaC_W().toString() << std::endl;
    std::cout << "nablaC_b: " << layer.getNablaC_b().toString() << std::endl;
    std::cout << "u:        " << layer.getU().toString() << std::endl;

    float sigmoid_prime_06 = 0.22878424045664325f;
    Vector<float> nablaC_b_expected = sigmoid_prime_06 * v;

    Matrix nablaC_W_expected = nablaC_b_expected.outer(x);
    Vector u_expected = Matrix<float>(2, 2, 0.5f).transposeMult(nablaC_b_expected);

    std::cout << "nablaC_W_expected: " << nablaC_W_expected.toString() << std::endl;
    std::cout << "nablaC_b_expected: " << nablaC_b_expected.toString() << std::endl;
    std::cout << "u_expected:        " << u_expected.toString() << std::endl;
}

TEST(LayerTest, update)
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

        std::cout << "W:    " << layer.getW().toString() << std::endl;
        std::cout << "b:    " << layer.getB().toString() << std::endl;
        std::cout << "Cost: " << 0.5f * v * v << std::endl;
        std::cout << "y:    " << layer.getY().toString() << std::endl;

        std::cout << std::endl;
    }
}

float sigmoid(float z)
{
    return 1.0f / (1.0f + exp(-z));
}

float sigmoid_prime(float z)
{
    float s = 1.0f / (1.0f + exp(-z));
    return (1.0f - s) * s;
}

Vector<float> sigmoid(sw::Vector<float> z)
{
    sw::Vector<float> y(z.size());

    for (int i = 0; i < z.size(); ++i)
    {
        y[i] = sigmoid(z[i]);
    }

    return y;
}

Vector<float> sigmoid_prime(sw::Vector<float> z)
{
    sw::Vector<float> y(z.size());

    for (int i = 0; i < z.size(); ++i)
    {
        y[i] = sigmoid_prime(z[i]);
    }

    return y;
}

TEST(LayerTest, notest)
{
    Matrix<float> W{{0.272f, -0.185f}, {0.735f, 1.204f}};
    Vector<float> b{-0.762f, 0.782f};
    Vector x{0.3f, 0.9f};

    Vector<float> z = W * x + b;
    std::cout << "y:    " << sigmoid(z).toString() << std::endl;
}

TEST(LayerTest, update2)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer = Layer(2, 2);

    Vector x1{0.3f, 0.9f};
    Vector x2{0.8f, 0.6f};

    Vector<float> v;

    for (int i = 0; i < 2000; i++)
    {

        layer.feedForward(x1);
        v = layer.getY() - x1; // u = dC / dx
        layer.backProp(x1, v);
        layer.update(1.0f);

        layer.feedForward(x2);
        v = layer.getY() - x2; // u = dC / dx
        layer.backProp(x2, v);
        layer.update(1.0f);
    }

    std::cout << "W:    " << layer.getW().toString() << std::endl;
    std::cout << "b:    " << layer.getB().toString() << std::endl;

    layer.feedForward(x1);
    std::cout << "Cost: " << 0.5f * v * v << std::endl;
    std::cout << "y:    " << layer.getY().toString() << std::endl;

    layer.feedForward(x2);
    std::cout << "Cost: " << 0.5f * v * v << std::endl;
    std::cout << "y:    " << layer.getY().toString() << std::endl;

    std::cout << std::endl;
}

TEST(LayerTest, 3to2Layer)
{
    // Create a network with default weights of 0.5 and bias of 0.
    Layer layer(3, 2);

    Vector x{0.3f, 0.6f, 0.9f};

    Vector<float> v;

    layer.feedForward(x);
    v = layer.getY() - x; // u = dC / dx
    layer.backProp(x, v);
    layer.update(1.0f);

    std::cout << "W:    " << layer.getW().toString() << std::endl;
    std::cout << "b:    " << layer.getB().toString() << std::endl;
    std::cout << "Cost: " << 0.5f * v * v << std::endl;
    std::cout << "y:    " << layer.getY().toString() << std::endl;

    std::cout << std::endl;
}

TEST(NetworkTest, network_one_input_vector)
{
    Network network(std::vector<uint16_t>{2, 2});

    Vector x1{0.3f, 0.9f};

    Vector<float> v;

    network.train(x1, x1);

    network.feedforward(x1);
    std::cout << network.getOutput().toString() << std::endl;
}

TEST(NetworkTest, network_two_intput_vectors)
{
    Network network(std::vector<uint16_t>{2, 2});

    Vector x1{0.3f, 0.9f};
    Vector x2{0.8f, 0.6f};

    Vector<float> v;

    for (int i = 0; i < 2000; i++)
    {
        network.train(x1, x1);
        network.train(x2, x2);
    }

    network.feedforward(x1);
    std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(x2);
    std::cout << network.getOutput().toString() << std::endl;
}

TEST(NetworkTest, multiLayeredNetwork)
{
    Network network(std::vector<uint16_t>{2, 2, 2});

    Vector x1{0.3f, 0.9f};
    Vector x2{0.8f, 0.6f};
    Vector x3{0.7f, 0.8f};

    for (int i = 0; i < 10000; ++i)
    {
        network.train(x1, x1);
        network.train(x2, x2);
        network.train(x3, x3);
    }

    network.feedforward(x1);
    std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(x2);
    std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(x3);
    std::cout << network.getOutput().toString() << std::endl;

    // TODO: make network return cost
}