#include <iostream>
#include <vector>
#include <chrono>

#include <random>
#include <algorithm>
#include <ranges>

#include "Net.hpp"
#include "Trainer.hpp"
#include "Evaluator.hpp"
// #include "Vector.hpp"
#include "Utils.hpp"
#include "Network.hpp"
#include "NetworkTrainer.hpp"
#include "MnistData.hpp"

int main()
{
    uint32_t nEpochs = 10;
    uint16_t miniBatchSize = 5;
    float learningRate = 3.0f;
    std::vector<uint16_t> layersizes{784, 30, 10};

    Network network(layersizes);
    network.randomizeWB(-2.0f, 2.0f);

    MnistData mnistData;
    mnistData.processMNIST();

    NetworkTrainer networkTrainer;
    networkTrainer.trainNet(network, nEpochs, learningRate, miniBatchSize, mnistData);

    printf("score: %d", networkTrainer.evalNet(network, mnistData));

    return 0;
}
