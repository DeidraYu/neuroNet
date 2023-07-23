#pragma once

#include <vector>
#include "Vector.hpp"
#include "Matrix.hpp"
#include "Net.hpp"
#include "Evaluator.hpp"

class Trainer
{
public:
    Trainer(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels);

    void train(Net &net, uint16_t nEpochs, uint16_t miniBatchSize, float learningRate, Evaluator evaluator);

private:
    struct ResultPair
    {
        std::vector<sw::Vector<float>> m_zs;
        std::vector<sw::Vector<float>> m_activations;

        ResultPair(const auto zs, const auto activations) : m_zs(zs), m_activations(activations) {}
    };

    struct GradientPair
    {
        std::vector<sw::Matrix<float>> m_weightGradient;
        std::vector<sw::Vector<float>> m_biasGradient;

        GradientPair(const auto weightGradient, const auto biasGradient) : m_weightGradient(weightGradient), m_biasGradient(biasGradient) {}
    };

    std::vector<std::vector<uint8_t>> &m_images;
    std::vector<uint8_t> &m_labels;

    ResultPair feedforward(Net &net, sw::VectorView<float> &image);
    GradientPair backProp(Net &net, ResultPair resultPair, uint8_t label, size_t minibatchSize, float learningRate);

    sw::Vector<float> oneHotEncode(int value, int numClasses);

    void trainEpoch(Net &net, uint16_t miniBatchSize, float learningRate, std::string progressLabel);
    void trainMiniBatch(Net &net, size_t imageIndex, size_t miniBatchSize, float learningRate);
};
