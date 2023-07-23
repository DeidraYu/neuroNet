#pragma once

#include "mnist/mnist_reader.hpp"

class MnistData
{
public:
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
    std::vector<sw::Vector<float>> train_data;
    std::vector<sw::Vector<float>> train_labels;

    std::vector<sw::Vector<float>> test_data;
    std::vector<sw::Vector<float>> test_labels;

    size_t numTrainImages;
    size_t numTestImages;

    void processMNIST();
};