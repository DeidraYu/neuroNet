#pragma once

#include <vector>
#include "MathUtils.hpp"
#include "Matrix.hpp"
#include "Net.hpp"

class Evaluator
{
public:
    Evaluator(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels);

    int evaluate(Net &net);

private:
    sw::Vector<float> feedforward(sw::VectorView<float> &image, Net &net);

    std::vector<std::vector<uint8_t>> &m_images;
    std::vector<uint8_t> &m_labels;
};