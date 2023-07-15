#pragma once

#include <vector>
#include "Vector.hpp"
#include "Matrix.hpp"
#include "Net.hpp"

class Trainer
{
public:
    Trainer(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels);

    void train(Net &net, uint16_t nEpochs, uint16_t miniBatchSize, float learningRate);

private:
    struct ResultPair
    {
        std::vector<sw::Vector<float>> zs;
        std::vector<sw::Vector<float>> activations;

        ResultPair(const auto zs, const auto activations) : zs(zs), activations(activations) {}
    };

    std::vector<std::vector<uint8_t>> &m_images;
    std::vector<uint8_t> &m_labels;

    ResultPair feedforward(Net &net, sw::VectorView<float> &image);

    void trainEpoch(Net &net, uint16_t miniBatchSize, float learningRate);
    void trainMiniBatch(Net &net, size_t imageIndex, size_t miniBatchSize, float learningRate);
};
