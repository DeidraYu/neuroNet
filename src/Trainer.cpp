#include <random>
#include <utility>

#include "Trainer.hpp"

Trainer::Trainer(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels) : m_images(images), m_labels(labels) {}

void Trainer::train(Net &net, uint16_t nEpochs, uint16_t miniBatchSize, float learningRate)
{
    // Setup random generator to permute the vector of images.
    std::random_device rd;
    std::mt19937 gen(rd());

    // Shuffle the training images and training labels
    for (uint16_t j = 0; j < nEpochs; ++j)
    {
        std::shuffle(m_images.begin(), m_images.end(), std::mt19937(gen));
        std::shuffle(m_labels.begin(), m_labels.end(), gen);

        trainEpoch(net, miniBatchSize, learningRate);
    }
}

void Trainer::trainEpoch(Net &net, uint16_t miniBatchSize, float learningRate)
{
    size_t numMiniBatches = (m_images.size() - 1) / miniBatchSize + 1;
    for (uint16_t miniBatchIndex = 0; miniBatchIndex < numMiniBatches; ++miniBatchIndex)
    {
        trainMiniBatch(net, miniBatchIndex, miniBatchSize, learningRate);
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
            trainMiniBatch(net, imageIndex, miniBatchSize, learningRate);
            imageIndex += miniBatchSize;
        }
        else
        {
            trainMiniBatch(net, imageIndex, m_images.size() - imageIndex, learningRate);
            imageIndex = m_images.size();
        }
    }
}

void Trainer::trainMiniBatch(Net &net, size_t imageIndex, size_t miniBatchSize, float learningRate)
{
    std::vector<sw::Matrix<float>> weightGradient(net.getSizes().size());
    std::vector<sw::Vector<float>> biasGradient(net.getSizes().size());

    for (size_t i = imageIndex; i < imageIndex + miniBatchSize; ++i)
    {
        sw::VectorView image(&m_images[i]);
        sw::Vector<float> image2 = image * (1.0f / 256.0f);
        uint8_t label = m_labels[i];

        ResultPair results = feedforward(net, image2);
        GradientPair gradients = backProp(net, results, label, miniBatchSize, learningRate);

        for (int i = 0; i < net.getSizes().size() - 1; i++)
        {
            weightGradient[i] = weightGradient[i] + gradients.weightGradient[i];
            biasGradient[i] = biasGradient[i] + gradients.biasGradient[i];
        }
    }
}

Trainer::ResultPair Trainer::feedforward(Net &net, sw::VectorView<float> &image)
{
    std::vector<uint16_t> layerSizes = net.getSizes();
    auto &weights = net.getWeights();
    auto &biases = net.getBiases();

    std::vector<sw::Vector<float>> activations;
    std::vector<sw::Vector<float>> zs;

    sw::Vector<float> z = weights[0] * image + biases[0];
    sw::Vector<float> activation = net.sigmoid(z);

    zs.push_back(z);
    activations.push_back(activation);

    for (int i = 1; i < layerSizes.size() - 1; ++i)
    {
        z = weights[i] * activation + biases[i];
        activation = net.sigmoid(z);

        zs.push_back(z);
        activations.push_back(activation);
    }

    return ResultPair(zs, activations);
}

Trainer::GradientPair Trainer::backProp(Net &net, ResultPair resultPair, uint8_t label, size_t miniBatchSize, float learningRate)
{
    sw::Vector<float> netWorkOutput = resultPair.activations[resultPair.activations.size() - 1];
    sw::Vector<float> errorVector = netWorkOutput - oneHotEncode(label, 10);

    auto weights = net.getWeights();
    auto biases = net.getBiases();

    size_t nLayers = net.getSizes().size();

    std::vector<sw::Matrix<float>> weightGradient(net.getSizes().size());
    std::vector<sw::Vector<float>> biasGradient(net.getSizes().size());

    sw::Vector<float> sigmoidPrime;
    sw::Vector<float> deltaZ = errorVector.point_mult(net.sigmoid_prime(resultPair.zs[nLayers - 1]));

    for (int i = 1; i < nLayers - 1; ++i)
    {

        sigmoidPrime = net.sigmoid_prime(resultPair.zs[nLayers - i]);
        deltaZ = sigmoidPrime.point_mult(weights[nLayers - i] * deltaZ);

        weightGradient[nLayers - i] = deltaZ.outer(resultPair.activations[nLayers - 1 - i]);
        biasGradient[nLayers - i] = deltaZ;
    }

    return GradientPair(weightGradient, biasGradient);
}

sw::Vector<float> Trainer::oneHotEncode(int value, int numClasses)
{
    sw::Vector<float> encodedVector(numClasses);
    if (value >= 0 && value < numClasses)
    {
        encodedVector[value] = 1.0f;
    }
    return encodedVector;
}
