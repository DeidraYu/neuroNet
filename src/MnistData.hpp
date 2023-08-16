#pragma once

#include "mnist/mnist_reader.hpp"
// #include "Vector.hpp"
#include "math/Vector.hpp"

using namespace sw::math;

class MnistData
{
public:
    MnistData() = default;

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
    std::vector<Vector<float>> train_data;
    std::vector<Vector<float>> train_labels;

    std::vector<Vector<float>> test_data;
    std::vector<Vector<float>> test_labels;

    size_t numTrainImages;
    size_t numTestImages;

    void processMNIST();
    void processMNIST(size_t trainingLimit, size_t testLimit);
};