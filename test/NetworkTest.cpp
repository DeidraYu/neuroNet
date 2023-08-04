#include <chrono>

#include "gtest/gtest.h"
#include "../src/Network.hpp"
#include "../src/NetworkTrainer.hpp"
#include "../src/MnistData.hpp"

class NetworkTest : public ::testing::Test
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

TEST_F(NetworkTest, network_one_input_vector)
{
    Network network(std::vector<uint16_t>{2, 2});

    Vector x1{0.3f, 0.9f};

    Vector<float> v;

    network.train(x1, x1);

    network.feedforward(x1);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;
}

TEST_F(NetworkTest, network_two_intput_vectors)
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
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(x2);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;
}

TEST_F(NetworkTest, multiLayeredNetwork)
{
    Network network(std::vector<uint16_t>{2, 2, 2});

    Vector x1{0.3f, 0.9f};
    Vector x2{0.8f, 0.6f};
    Vector x3{0.7f, 0.8f};

    for (int i = 0; i < 100; ++i)
    {
        network.train(x1, x1);
        network.train(x2, x2);
        network.train(x3, x3);
    }

    network.feedforward(x1);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(x2);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(x3);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    // TODO: make network return cost
}

TEST_F(NetworkTest, trainNet)
{
    uint16_t nEpochs = 1000;
    uint16_t miniBatchSize = 2;
    float learningRate = 1.0f;
    std::vector<uint16_t> layersizes{2, 2, 2};

    std::vector<Vector<float>> train_data{{0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 0.5f}, {0.0f, 1.0f}};
    std::vector<Vector<float>> train_labels{{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}};

    Network network(layersizes);

    for (int i = 0; i < nEpochs; ++i)
    {
        network.trainMiniBatches(miniBatchSize, learningRate, train_data, train_labels);
    }

    network.feedforward(train_data[0]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(train_data[1]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(train_data[2]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(train_data[3]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;
    // TODO: replace with evaluator
}

TEST_F(NetworkTest, trainNetRandomWB)
{
    uint16_t nEpochs = 1000;
    uint16_t miniBatchSize = 2;
    float learningRate = 1.0f;
    std::vector<uint16_t> layersizes{2, 5, 2};

    std::vector<Vector<float>> train_data{{0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 0.5f}, {0.0f, 1.0f}};
    std::vector<Vector<float>> train_labels{{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}};

    Network network(layersizes);
    network.randomizeWB(-2.0f, 2.0f);

    for (int i = 0; i < nEpochs; ++i)
    {
        network.trainMiniBatches(miniBatchSize, learningRate, train_data, train_labels);
    }

    network.feedforward(train_data[0]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(train_data[1]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(train_data[2]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;

    network.feedforward(train_data[3]);
    if (sw::verbosity)
        std::cout << network.getOutput().toString() << std::endl;
    // TODO: replace with evaluator
}

TEST_F(NetworkTest, onSubsetOfMnistData)
{
    uint32_t nEpochs = 2;
    uint16_t miniBatchSize = 5;
    float learningRate = 10.0f;
    std::vector<uint16_t> layersizes{784, 30, 10};

    // Read MNIST dataset and convert to our own Vector format
    MnistData mnistData;
    mnistData.processMNIST(600, 100);

    // Setup the network topology
    Network network(layersizes);

    // Create the network trainer wrapper
    NetworkTrainer networkTrainer;

    Random::seed(0);
    network.randomizeWB(-2.0f, 2.0f);

    networkTrainer.trainNet(network, nEpochs, learningRate, miniBatchSize, mnistData);
    uint32_t score = networkTrainer.evalNet(network, mnistData);

    EXPECT_EQ(score, 50); // out of 100
}
