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

    struct GradientPair
    {
        std::vector<sw::Matrix<float>> weightGradient;
        std::vector<sw::Vector<float>> biasGradient;

        GradientPair(const auto weightGradient, const auto biasGradient) : weightGradient(weightGradient), biasGradient(biasGradient) {}
    };

    std::vector<std::vector<uint8_t>> &m_images;
    std::vector<uint8_t> &m_labels;

    ResultPair feedforward(Net &net, sw::VectorView<float> &image);
    GradientPair backProp(Net &net, ResultPair resultPair, uint8_t label, size_t minibatchSize, float learningRate);

    sw::Vector<float> oneHotEncode(int value, int numClasses);

    void trainEpoch(Net &net, uint16_t miniBatchSize, float learningRate, std::string progressLabel);
    void trainMiniBatch(Net &net, size_t imageIndex, size_t miniBatchSize, float learningRate);
};
