#include "Evaluator.hpp"
#include "Utils.hpp"

Evaluator::Evaluator(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels) : m_images(images), m_labels(labels) {}

int Evaluator::evaluate(Net &net)
{
    printNumber(m_images[0]);
    sw::VectorView image(&m_images[0]);
    sw::Vector<float> image2 = image * (1.0f / 256.0f);
    feedforward(image2, net);

    return 0;
}

sw::Vector<float> Evaluator::feedforward(sw::VectorView<float> &image, Net &net)
{
    std::vector<uint16_t> layerSizes = net.getSizes();
    auto &weights = net.getWeights();
    auto &biases = net.getBiases();
    std::vector<sw::Vector<float>> activations(weights.size() + 1);

    activations[0] = sw::Vector<float>(layerSizes[0]);

    for (int i = 1; i < layerSizes.size(); ++i)
    {
        net.sigmoid(weights[i] * activations[i - 1] + biases[i]);
    }

    return sw::Vector<float>(3);
}