#include <iostream>
#include <vector>
#include <chrono>

#include <random>
#include <algorithm>
#include <ranges>

#include "Utils.hpp"
#include "Network.hpp"
#include "NetworkTrainer.hpp"
#include "MnistData.hpp"
#include "math/utils.hpp"

int main()
{
    // sw::verbosity = 0; // suppress progress bar

    uint32_t nEpochs = 10;
    uint16_t miniBatchSize = 10;
    float learningRate = 1.0f;
    std::vector<uint16_t> layersizes{784, 30, 10};

    Network network(layersizes);
    Random::seed(); // Use a randomized seed
    network.randomizeWB(-2.0f, 2.0f);

    MnistData mnistData;
    mnistData.processMNIST();

    NetworkTrainer networkTrainer;
    networkTrainer.trainNet(network, nEpochs, learningRate, miniBatchSize, mnistData);

    std::cout << sw::prof::Times::toString(nEpochs) << std::endl;

    uint32_t score = networkTrainer.evalNet(network, mnistData);

    printf("\nscore: %d\n\n", score);

    return 0;
}
