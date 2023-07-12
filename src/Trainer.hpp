#pragma once

#include <vector>

class Trainer
{
public:
    Trainer(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels);

    void train(uint16_t nEpochs, uint16_t miniBatchSize, float learningRate);

private:
    std::vector<std::vector<uint8_t>> &m_images;
    std::vector<uint8_t> &m_labels;

    void trainEpoch(uint16_t miniBatchSize, float learningRate);
    void trainMiniBatch(uint16_t imageIndex, uint16_t miniBatchSize, float learningRate);
};
