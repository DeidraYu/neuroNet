#pragma once

#include <cstdint>
#include <vector>
#include "Vector.hpp"
#include "Matrix.hpp"
#include "Layer.hpp"
#include "Utils.hpp"

class Network
{
public:
    Network(std::vector<uint16_t> layerSizes) : m_layerSizes(layerSizes)
    {
        for (int k = 0; k < m_layerSizes.size() - 1; ++k)
        {
            m_layers.push_back(Layer(layerSizes[k], layerSizes[k + 1]));
        }
    }

    void trainMiniBatches(uint16_t miniBatchSize, float learningRate, std::vector<sw::Vector<float>> &train_data, std::vector<sw::Vector<float>> &train_labels)
    {
        uint16_t mbIndex = 0;
        m_eta = learningRate;

        for (uint16_t i = 0; i < train_data.size(); ++i)
        {
            train(train_data[i], train_labels[i]);

            if (mbIndex == miniBatchSize - 1)
            {
                mbIndex = 0;
                updateLayers();
            }
            else
            {
                ++mbIndex;
            }
        }
    }

    void train(const Vector<float> &x, sw::Vector<float> label)
    {
        // Feed forward run
        feedforward(x);

        // Compute the error vector
        Vector<float> v = getOutput() - label;

        // Back propagation run

        // L is index of last layer:
        int L = static_cast<int>(m_layers.size()) - 1; // k = 0, 1, ..., L  (so, L is inclusive)

        if (L == 0) // there is only one layer
        {
            m_layers.back().backProp(x, v);
        }
        else
        {
            m_layers.back().backProp(m_layers[L - 1].getY(), v);
            for (int k = L - 1; k > 0; --k)
            {
                m_layers[k].backProp(m_layers[k - 1].getY(), m_layers[k + 1].getU());
            }
            m_layers[0].backProp(x, m_layers[1].getU());
        }
    }

    void feedforward(sw::Vector<float> x)
    {
        m_layers[0].feedForward(x);
        for (int k = 1; k < m_layers.size(); ++k)
        {
            m_layers[k].feedForward(m_layers[k - 1].getY());
        }
    }

    void updateLayers()
    {
        // update
        for (int k = 0; k < m_layers.size(); ++k)
        {
            m_layers[k].update(m_eta);
        }
    }

    sw::Vector<float> getOutput()
    {
        return m_layers.back().getY();
    }

private:
    float m_eta = 1.0f;

    std::vector<Layer> m_layers;
    std::vector<uint16_t> m_layerSizes;
};