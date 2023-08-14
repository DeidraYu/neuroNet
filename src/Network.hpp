#pragma once

#include <cstdint>
#include <vector>
// #include "Vector.hpp"
// #include "Matrix.hpp"
#include "math/Vector.hpp"
#include "math/Matrix.hpp"
#include "Layer.hpp"
#include "Utils.hpp"

using namespace sw::math;

class Network
{
public:
    Network() : m_layerSizes(std::vector<uint16_t>()) {}

    Network(std::vector<uint16_t> layerSizes) : m_layerSizes(layerSizes)
    {
        for (uint32_t k = 0; k < m_layerSizes.size() - 1; ++k)
        {
            ActivationType activationType = ActivationType::Sigmoid;
            if (k < m_layerSizes.size() - 2)
            {
                activationType = ActivationType::Relu;
            }
            Layer newLayer(layerSizes[k], layerSizes[k + 1], activationType);
            m_layers.push_back(newLayer);
        }
    }

    void trainMiniBatches(uint16_t miniBatchSize, float learningRate, std::vector<Vector<float>> &train_data, std::vector<Vector<float>> &train_labels)
    {
        sw::prof::Measure M("networkTrainer.trainEpoch");

        uint16_t mbIndex = 0;
        m_eta = learningRate;
        int numMiniBatches = static_cast<int>((train_data.size() - 1) / miniBatchSize + 1);
        int miniBatchCounter = 0;

        for (uint16_t i = 0; i < train_data.size(); ++i)
        {
            train(train_data[i], train_labels[i]);

            if (mbIndex == miniBatchSize - 1)
            {
                mbIndex = 0;
                updateLayers();
                printProgress(miniBatchCounter, numMiniBatches, progressLabel);
                ++miniBatchCounter;
            }
            else
            {
                ++mbIndex;
            }
        }

        if (miniBatchCounter != numMiniBatches)
        {
            updateLayers();
            printProgress(miniBatchCounter, numMiniBatches, progressLabel);
        }
    }

    void train(const Vector<float> &x, const Vector<float> &label)
    {
        sw::prof::Measure M("NetworkTrainer.train");

        feedforward(x);
        Vector<float> v = computeLoss(label);
        backProp(x, v);
    }

    void feedforward(const Vector<float> &x)
    {
        // sw::prof::Measure M("Network::feedForward");

        m_layers[0].feedForward(x);
        for (uint32_t k = 1; k < m_layers.size(); ++k)
        {
            m_layers[k].feedForward(m_layers[k - 1].getY());
        }
    }

    void fastFeedforward(const Vector<float> &x)
    {
        // sw::prof::Measure M("Network::fastFeedForward");

        m_layers[0].fastFeedForward(x);
        for (uint32_t k = 1; k < m_layers.size(); ++k)
        {
            m_layers[k].fastFeedForward(m_layers[k - 1].getY());
        }
    }

    Vector<float> computeLoss(const Vector<float> &label)
    {
        // sw::prof::Measure M("Network::computeLoss");
        return std::move(getOutput() - label);
    }

    void backProp(const Vector<float> &x, const Vector<float> &v)
    {
        // sw::prof::Measure M("Network::backProp");

        // L is index of last layer:
        int L = static_cast<int>(m_layers.size()) - 1; // k = 0, 1, ..., L  (so, L is inclusive)

        if (L == 0) // there is only one layer
        {
            m_layers.back().backProp(x, v, false);
        }
        else
        {
            m_layers.back().backProp(m_layers[L - 1].getY(), v);
            for (int k = L - 1; k > 0; --k)
            {
                m_layers[k].backProp(m_layers[k - 1].getY(), m_layers[k + 1].getU());
            }
            m_layers[0].backProp(x, m_layers[1].getU(), false);
        }
    }

    void updateLayers()
    {
        sw::prof::Measure M("NetworkTrainer.updateLayers");

        // update
        for (uint32_t k = 0; k < m_layers.size(); ++k)
        {
            m_layers[k].update(m_eta);
        }
    }

    void addLayer(Layer layer)
    {
        m_layers.push_back(layer);
    }

    Vector<float> getOutput()
    {
        return m_layers.back().getY();
    }

    std::vector<Layer> &getLayers()
    {
        return m_layers;
    }

    size_t getNumLayers()
    {
        return m_layerSizes.size();
    }

    void randomizeWB(float min, float max)
    {
        for (Layer &layer : m_layers)
        {
            layer.randomizeWB(min, max);
        }
    }

    std::string progressLabel;

private:
    float m_eta = 1.0f;

    std::vector<Layer> m_layers;
    std::vector<uint16_t> m_layerSizes;
};