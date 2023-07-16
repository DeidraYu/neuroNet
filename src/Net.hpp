#pragma once

#include <cstdint>
#include <vector>
#include "Vector.hpp"
#include "Matrix.hpp"

class Net
{
public:
    Net(std::vector<uint16_t> layerSizes);

    void printInfo();
    std::vector<sw::Matrix<float>> &getWeights();
    std::vector<sw::Vector<float>> &getBiases();

    void updateWeights(std::vector<sw::Matrix<float>>);
    void updateBiases(std::vector<sw::Vector<float>>);
    std::vector<uint16_t> getSizes();

    sw::Vector<float> sigmoid(sw::Vector<float> z);
    sw::Vector<float> sigmoid_prime(sw::Vector<float> z);

private:
    std::vector<uint16_t> m_layerSizes;
    std::vector<sw::Matrix<float>> m_weights;
    std::vector<sw::Vector<float>> m_biases;

    void randomizeWeightsBiases();
};