#include <random>

#include "Trainer.hpp"

Trainer::Trainer(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels) : m_images(images), m_labels(labels) {}

void Trainer::train(uint16_t nEpochs, uint16_t miniBatchSize, float learningRate)
{
    // Setup random generator to permute the vector of images.
    std::random_device rd;
    std::mt19937 gen(rd());

    // Shuffle the training images and training labels
    for (uint16_t j = 0; j < nEpochs; ++j)
    {
        std::shuffle(m_images.begin(), m_images.end(), std::mt19937(gen));
        std::shuffle(m_labels.begin(), m_labels.end(), gen);

        trainEpoch(miniBatchSize, learningRate);
    }
}

void Trainer::trainEpoch(uint16_t miniBatchSize, float learningRate)
{
    size_t numMiniBatches = (m_images.size() - 1) / miniBatchSize + 1;
    for (uint16_t miniBatchIndex = 0; miniBatchIndex < numMiniBatches; ++miniBatchIndex)
    {
        trainMiniBatch(miniBatchIndex, miniBatchSize, learningRate);
    }

    // create the first index of the mini batch and the number of items in the mini batch:
    // For example, assume there are 11 images and the mini batch size is 10
    // 0 10
    // 10 1
    size_t imageIndex = 0;
    while (imageIndex < m_images.size())
    {
        if (imageIndex + miniBatchSize <= m_images.size())
        {
            trainMiniBatch(imageIndex, miniBatchSize, learningRate);
            imageIndex += miniBatchSize;
        }
        else
        {
            trainMiniBatch(imageIndex, m_images.size() - imageIndex, learningRate);
            imageIndex = m_images.size();
        }
    }
}

void Trainer::trainMiniBatch(uint16_t imageIndex, uint16_t miniBatchSize, float learningRate)
{
    for (size_t i = imageIndex; i < imageIndex + miniBatchSize; ++i)
    {
        std::vector<uint8_t> image = m_images[i];
        uint8_t label = m_labels[i];
    }
}
