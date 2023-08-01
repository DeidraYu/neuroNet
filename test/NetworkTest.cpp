#include <chrono>

#include "gtest/gtest.h"
#include "../src/Network.hpp"
#include "../src/NetworkTrainer.hpp"
#include "../src/MnistData.hpp"

// using namespace sw::math;

TEST(NetworkTest, test_01)
{
    uint32_t nEpochs = 2;
    uint16_t miniBatchSize = 10;
    float learningRate = 1.0f;
    std::vector<uint16_t> layersizes{784, 30, 10};

    // Read MNIST dataset and convert to our own Vector format
    MnistData mnistData;
    mnistData.processMNIST();

    // Setup the network topology
    Network network(layersizes);

    // Create the network trainer wrapper
    NetworkTrainer networkTrainer;

    Random::seed(0);
    network.randomizeWB(-2.0f, 2.0f);

    networkTrainer.trainNet(network, nEpochs, learningRate, miniBatchSize, mnistData);
    uint32_t score = networkTrainer.evalNet(network, mnistData);

    EXPECT_EQ(score, 8944);
}
