#include <iostream>
#include <vector>
#include <chrono>

#include <random>
#include <algorithm>
#include <ranges>

#include "mnist/mnist_reader.hpp"
#include "Net.hpp"
#include "Trainer.hpp"
#include "Evaluator.hpp"
#include "Vector.hpp"
#include "Utils.hpp"

int main()
{

    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    auto start1 = std::chrono::steady_clock::now();

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    auto end1 = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    std::cout << "Time reading mnist dataset from disk: " << duration1.count() << " milliseconds" << std::endl;

    if (false)
    {
        printNumber(dataset.training_images[0]);
        printNumber(dataset.training_images[1]);
        printNumber(dataset.training_images[2]);
    }

    Net net({28 * 28, 196, 49, 10});
    // net.printInfo();

    Trainer trainer(dataset.training_images, dataset.training_labels);

    {
        auto start2 = std::chrono::steady_clock::now();

        int16_t nEpochs = 10;
        uint16_t miniBatchSize = 5;
        float learningRate = 2.0f;

        trainer.train(net, nEpochs, miniBatchSize, learningRate);

        auto end2 = std::chrono::steady_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
        printf("Time permuting data set %d times: %lld milliseconds\n", nEpochs, duration2.count());
    }

    Evaluator evaluator(dataset.test_images, dataset.test_labels);

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

    return 0;
}
