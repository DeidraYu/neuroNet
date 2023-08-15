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
#include "FileIO.hpp"

void printRunInfo(uint32_t nEpochs, uint16_t miniBatchSize, float learningRate, std::vector<uint16_t> layersizes, uint32_t seed)
{
    std::string layerSizesOutput = "(";
    layerSizesOutput += std::to_string(layersizes[0]);
    for (size_t i = 1; i < layersizes.size(); ++i)
        layerSizesOutput += ", " + std::to_string(layersizes[i]);
    layerSizesOutput += ")";

    // Example output: ./neuronet 100 10 1 "(784, 30, 10)" 2964684087
    std::cout << "./neuronet"
              << " " << nEpochs << " " << miniBatchSize << " " << learningRate << " " << '"' << layerSizesOutput << '"' << " " << seed << std::endl;
    std::cout << "nEpochs: " << nEpochs << ",  miniBatchSize: " << miniBatchSize << ",  learningRate: " << learningRate << ", layerSizes " << layerSizesOutput << ",  seed: " << seed << std::endl;
}

int main(int argc, char *argv[])
{
    uint32_t nEpochs = 10;
    uint16_t miniBatchSize = 10;
    float learningRate = 1.0f;
    std::vector<uint16_t> layersizes{784, 30, 10};
    uint32_t seed = Random::seed(); // Use a randomized seed

    if (argc >= 2)
    {
        nEpochs = std::stoi(argv[1]); // Convert to uint32_t
    }

    if (argc >= 3)
    {
        miniBatchSize = std::stoi(argv[2]); // Convert to uint16_t
    }

    if (argc >= 4)
    {
        learningRate = std::stof(argv[3]); // Convert to float
    }

    // Extract layersizes input from the command-line argument
    if (argc >= 5)
    {
        std::string layersizesInput = argv[4];
        layersizes.clear(); // Clear the default layersizes

        // Remove the leading and trailing parentheses, if any
        if (layersizesInput.front() == '(')
            layersizesInput.erase(0, 1);
        if (layersizesInput.back() == ')')
            layersizesInput.pop_back();

        // Convert the layersizes input to a stringstream
        std::stringstream ss(layersizesInput);

        // Split the input string using ',' as a delimiter and add to layersizes vector
        uint16_t size;
        while (ss >> size)
        {
            layersizes.push_back(size);
            if (ss.peek() == ',')
                ss.ignore();
        }
    }

    if (argc >= 6)
    {
        int seedIn = std::strtoul(argv[5], nullptr, 10);
        seed = Random::seed(seedIn);
    }

    if (argc >= 7)
    {
        sw::verbosity = std::stoi(argv[6]); // Convert to uint16_t
    }

    Network network(layersizes);
    // network.randomizeWB(-2.0f, 2.0f);
    network.randomizeWB(-0.0001f, 0.0001f); // ./neuronet 10 10 1 "(784, 30, 10)" 1345823770

    MnistData mnistData;
    mnistData.processMNIST();

    NetworkTrainer networkTrainer;
    printRunInfo(nEpochs, miniBatchSize, learningRate, layersizes, seed);
    networkTrainer.trainNet(network, nEpochs, learningRate, miniBatchSize, mnistData);

    std::cout << sw::prof::Times::toString(nEpochs) << std::endl;

    uint32_t score = networkTrainer.evalNet(network, mnistData);

    printf("\nscore: %d\n\n", score);

    // FileIO fileIO;
    // fileIO.saveNetwork(network, std::string("/home/koenr/projects/neuroNetSaves/testSave.bin"));

    return 0;
}
