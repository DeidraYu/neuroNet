#include <iostream>
#include <vector>
#include <chrono>

#include <random>
#include <algorithm>
#include <ranges>

#include "Net.hpp"
#include "Trainer.hpp"
#include "Evaluator.hpp"
#include "Vector.hpp"
#include "Utils.hpp"
#include "Network.hpp"
#include "NetworkTrainer.hpp"
#include "MnistData.hpp"

int main()
{
    uint32_t nEpochs = 5;
    uint32_t miniBatchSize = 10;
    float learningRate = 3.0f;
    std::vector<uint16_t> layersizes{784, 30, 10};

    Network network(layersizes);
    network.randomizeWB(-2.0f, 2.0f);

    MnistData mnistData;
    mnistData.processMNIST();

    NetworkTrainer networkTrainer;
    networkTrainer.trainNet(network, nEpochs, learningRate, miniBatchSize, mnistData);

    printf("score: %d", networkTrainer.evalNet(network, mnistData));

    /*
    if (false)
    {
        printNumber(dataset.training_images[0]);
        printNumber(dataset.training_images[1]);
        printNumber(dataset.training_images[2]);
    }


        Net net({28 * 28, 30, 10});
        // net.printInfo();

        Evaluator evaluator(dataset.test_images, dataset.test_labels);
        Trainer trainer(dataset.training_images, dataset.training_labels);

        {
            auto start2 = std::chrono::steady_clock::now();

            int16_t nEpochs = 3;
            uint16_t miniBatchSize = 5;
            float learningRate = 2.0f;

            trainer.train(net, nEpochs, miniBatchSize, learningRate, evaluator);

            auto end2 = std::chrono::steady_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
            printf("Time permuting data set %d times: %lld milliseconds\n", nEpochs, duration2.count());
        }

        int score = 0;
        int n = 1;

        auto start3 = std::chrono::steady_clock::now();

        for (int i = 0; i < n; ++i)
        {
            score += evaluator.evaluate(net);
        }

        auto end3 = std::chrono::steady_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
        printf("Time evaluating %d times: %lld ms\n", n, duration3.count());
        printf("Eval took: %lld ms\n", evaluator.duration / 1000);
        printf("Score: %8.4f%%\n", (score * 100.0f) / dataset.test_images.size());
    */

    return 0;
}

void MnistData::processMNIST()
{
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    auto start1 = std::chrono::steady_clock::now();

    // Load MNIST data
    dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    auto end1 = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    std::cout << "Time reading mnist dataset from disk: " << duration1.count() << " milliseconds" << std::endl;

    numTrainImages = dataset.training_images.size();
    numTestImages = dataset.test_images.size();

    train_data = std::vector<sw::Vector<float>>(numTrainImages);
    train_labels = std::vector<sw::Vector<float>>(numTrainImages);

    test_data = std::vector<sw::Vector<float>>(numTestImages);
    test_labels = std::vector<sw::Vector<float>>(numTestImages);

    {
        for (uint32_t i = 0; i < numTrainImages; ++i)
        {
            sw::VectorView train_image_int(&(dataset.training_images[i]));
            sw::Vector<float> train_image = train_image_int * (1.0f / 256.0f);

            train_data[i] = std::move(train_image);

            train_labels[i] = oneHotEncode(dataset.training_labels[i], 10);
        }

        for (uint32_t i = 0; i < numTestImages; ++i)
        {
            sw::VectorView test_image_int(&(dataset.test_images[i]));
            sw::Vector<float> test_image = test_image_int * (1.0f / 256.0f);

            test_data[i] = std::move(test_image);

            test_labels[i] = oneHotEncode(dataset.test_labels[i], 10);
        }
    }
}
