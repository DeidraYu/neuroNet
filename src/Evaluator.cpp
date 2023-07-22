#include <iostream>
#include <algorithm>
#include <chrono>

#include "Evaluator.hpp"
#include "Utils.hpp"

Evaluator::Evaluator(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels) : m_images(images), m_labels(labels) {}

int Evaluator::evaluate(Net &net)
{
    int score = 0;

    // printNumber(m_images[0]);
    /*sw::VectorView image(&m_images[0]);
    sw::Vector<float> image2 = image * (1.0f / 256.0f);
    sw::Vector<float> a = feedforward(image2, net);*/

    // printf("%s\n", a.toString().c_str());
    // printf("%d\n", a.argmax());

    for (int i = 0; i < m_images.size(); ++i)
    {
        sw::VectorView image(&m_images[i]);
        sw::Vector<float> image2 = image * (1.0f / 256.0f);
        sw::Vector<float> a = feedforward(image2, net);

        if (a.argmax() == m_labels[i])
        {
            ++score;
        }
    }

    return score;
}

sw::Vector<float> Evaluator::feedforward(sw::VectorView<float> &image, Net &net)
{
    std::vector<uint16_t> layerSizes = net.getSizes();
    auto &weights = net.getWeights();
    auto &biases = net.getBiases();
    sw::Vector<float> activation;

    activation = sw::Vector<float>(layerSizes[0]);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < layerSizes.size() - 1; ++i)
    {
        activation = net.sigmoid(weights[i].transposeMult(activation) + biases[i]);
    }

    auto end = std::chrono::steady_clock::now();
    duration = duration + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return activation;
}
