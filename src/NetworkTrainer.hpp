#pragma once

// #include <iostream>
// #include <vector>
// #include <chrono>

// #include <random>
// #include <algorithm>
// #include <ranges>

// #include "Net.hpp"
// #include "Trainer.hpp"
// #include "Evaluator.hpp"
// #include "Vector.hpp"
// #include "Utils.hpp"
#include "Network.hpp"
#include "MnistData.hpp"

class NetworkTrainer
{
public:
    void trainNet(Network &network, uint32_t nEpochs, float learningRate, uint16_t miniBatchSize, MnistData &mnistData);

    uint32_t evalNet(Network &network, MnistData &mnistData);
};