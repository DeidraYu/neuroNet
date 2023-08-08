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

int main(int argc, char *argv[])
{
    uint32_t nEpochs = 10;
    uint16_t miniBatchSize = 10;
    float learningRate = 1.0f;
    std::vector<uint16_t> layersizes{784, 30, 10};

    if (argc >= 4)
    {
        nEpochs = std::stoi(argv[1]);       // Convert to uint32_t
        miniBatchSize = std::stoi(argv[2]); // Convert to uint16_t
        learningRate = std::stof(argv[3]);  // Convert to float

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
    }

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

    // FileIO fileIO;
    // fileIO.saveNetwork(network, std::string("/home/koenr/projects/neuroNetSaves/testSave.bin"));

    return 0;
}
