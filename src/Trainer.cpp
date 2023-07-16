#include <random>
#include <utility>

#include "Trainer.hpp"
#include "Utils.hpp"

Trainer::Trainer(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels) : m_images(images), m_labels(labels) {}

void Trainer::train(Net &net, uint16_t nEpochs, uint16_t miniBatchSize, float learningRate)
{
    // Setup random generator to permute the vector of images.
    std::random_device rd;
    std::mt19937 gen(rd());
    char progressLabel[16];
    // Shuffle the training images and training labels
    for (uint16_t j = 0; j < nEpochs; ++j)
    {
        std::shuffle(m_images.begin(), m_images.end(), std::mt19937(gen));
        std::shuffle(m_labels.begin(), m_labels.end(), gen);
        snprintf(progressLabel, 16, "%d / %d", j + 1, nEpochs);
        trainEpoch(net, miniBatchSize, learningRate, progressLabel);
    }
    printf("\n"); // because the progress bar has no newline.
}

void Trainer::trainEpoch(Net &net, uint16_t miniBatchSize, float learningRate, std::string progressLabel)
{
    int numMiniBatches = static_cast<int>((m_images.size() - 1) / miniBatchSize + 1);
    // for (uint16_t miniBatchIndex = 0; miniBatchIndex < numMiniBatches; ++miniBatchIndex)
    // {
    //     trainMiniBatch(net, miniBatchIndex, miniBatchSize, learningRate);
    // }

    // create the first index of the mini batch and the number of items in the mini batch:
    // For example, assume there are 11 images and the mini batch size is 10
    // 0 10
    // 10 1
    size_t imageIndex = 0;
    int miniBatchCounter = 0;

    while (imageIndex < m_images.size())
    {
        printProgress(miniBatchCounter, numMiniBatches, progressLabel);

        if (imageIndex + miniBatchSize <= m_images.size())
        {
            trainMiniBatch(net, imageIndex, miniBatchSize, learningRate);
            imageIndex += miniBatchSize;
        }
        else
        {
            trainMiniBatch(net, imageIndex, m_images.size() - imageIndex, learningRate);
            imageIndex = m_images.size();
        }
        miniBatchCounter++;
    }
    printProgress(numMiniBatches, numMiniBatches, progressLabel);
}

void Trainer::trainMiniBatch(Net &net, size_t imageIndex, size_t miniBatchSize, float learningRate)
{
    for (size_t i = imageIndex; i < imageIndex + miniBatchSize; ++i)
    {
        sw::VectorView image(&m_images[i]);
        sw::Vector<float> image2 = image * (1.0f / 256.0f);
        uint8_t label = m_labels[i];

        ResultPair results = feedforward(net, image2);
    }
}

Trainer::ResultPair Trainer::feedforward(Net &net, sw::VectorView<float> &image)
{
    std::vector<uint16_t> layerSizes = net.getSizes();
    auto &weights = net.getWeights();
    auto &biases = net.getBiases();
    sw::Vector<float> activation;
    sw::Vector<float> z;

    std::vector<sw::Vector<float>> activations;
    std::vector<sw::Vector<float>> zs;

    activation = sw::Vector<float>(layerSizes[0]);
    z = sw::Vector<float>(layerSizes[0]);

    for (int i = 0; i < layerSizes.size() - 1; ++i)
    {
        z = weights[i] * activation + biases[i];
        activation = net.sigmoid(z);

        zs.push_back(z);
        activations.push_back(activation);
    }

    return ResultPair(zs, activations);
}
