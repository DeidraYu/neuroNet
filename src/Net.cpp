#include <cstdio>
#include <vector>
#include <random>

#include "Net.hpp"

Net::Net(std::vector<uint16_t> layerSizes) : m_layerSizes(layerSizes), m_weights(layerSizes.size() - 1), m_biases(layerSizes.size() - 1)
{
    for (int layerIndex = 0; layerIndex < layerSizes.size() - 1; ++layerIndex)
    {
        m_weights[layerIndex] = std::move(sw::Matrix<float>(layerSizes[layerIndex + 1], layerSizes[layerIndex]));
        m_biases[layerIndex] = std::move(sw::Vector<float>(layerSizes[layerIndex + 1]));
    }

    randomizeWeightsBiases();
}

void Net::printInfo()
{
    printf("Number of nodes per layer:\n");
    for (size_t layerIndex = 0; layerIndex < m_layerSizes.size(); ++layerIndex)
    {
        printf("    %2lld: %d\n", layerIndex, m_layerSizes[layerIndex]);
    }
}

void Net::randomizeWeightsBiases()
{
    float min = -2.0f;
    float max = 2.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (int n = 0; n < m_weights.size(); ++n)
    {
        for (uint32_t i = 0; i < m_weights[n].getNumRows(); ++i)
        {
            for (uint32_t j = 0; j < m_weights[n].getNumCols(); ++j)
            {
                m_weights[n][i][j] = dis(gen);
            }
        }
    }

    for (int n = 0; n < m_biases.size(); ++n)
    {
        for (int i = 0; i < m_biases[n].size(); ++i)
        {
            m_biases[n][i] = dis(gen);
        }
    }
}

sw::Vector<float> Net::sigmoid(sw::Vector<float> z)
{
    sw::Vector<float> activations(z.size());

    for (int i = 0; i < z.size() - 1; ++i)
    {
        activations[i] = 1.0f / (1.0f + exp(-z[i]));
    }

    return activations;
}

std::vector<sw::Matrix<float>> &Net::getWeights()
{
    // std::vector<sw::Matrix<float>> test{sw::Matrix<float>(1, 1)};
    return m_weights;
}

std::vector<sw::Vector<float>> &Net::getBiases()
{
    // std::vector<sw::Vector<float>> test{sw::Vector<float>(1)};
    return m_biases;
}

std::vector<uint16_t> Net::getSizes()
{
    return m_layerSizes;
}
