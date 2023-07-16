#include <random>
#include <utility>
#include <chrono>

#include "Trainer.hpp"
#include "Utils.hpp"

Trainer::Trainer(std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels) : m_images(images), m_labels(labels) {}

void Trainer::train(Net &net, uint16_t nEpochs, uint16_t miniBatchSize, float learningRate, Evaluator evaluator)
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

        printf("\n Score after %d epochs: %d / %d", j + 1, evaluator.evaluate(net), 10000);
    }
    printf("\n"); // because the progress bar has no newline.
}

void Trainer::trainEpoch(Net &net, uint16_t miniBatchSize, float learningRate, std::string progressLabel)
{
    printf(net.getBiases()[1].toString().c_str());
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
    // auto start = std::chrono::steady_clock::now();

    std::vector<sw::Matrix<float>> weightGradient(net.getSizes().size() - 1);
    std::vector<sw::Vector<float>> biasGradient(net.getSizes().size() - 1);
    std::vector<uint16_t> netSizes = net.getSizes();

    for (int j = 0; j < netSizes.size() - 1; ++j)
    {

        weightGradient[j] = sw::Matrix(netSizes[j + 1], netSizes[j], 0.0f);
        biasGradient[j] = sw::Vector(netSizes[j + 1], 0.0f);
    }

    for (size_t i = imageIndex; i < imageIndex + miniBatchSize; ++i)
    {
        sw::VectorView image(&m_images[i]);
        sw::Vector<float> image2 = image * (1.0f / 256.0f);
        uint8_t label = m_labels[i];

        ResultPair results = feedforward(net, image2);
        GradientPair gradients = backProp(net, results, label, miniBatchSize, learningRate);

        // weightGradient[0] = gradients.weightGradient[0];
        // biasGradient[0] = gradients.biasGradient[0];

        for (int j = 0; j < netSizes.size() - 1; j++)
        {

            // !!!!!!!!!!!!!!!!!!! make it += this is a temporary fix !!!!!!!!!!!!!!!!!!!!!!!!!!
            weightGradient[j] = weightGradient[j] + gradients.weightGradient[j];
            biasGradient[j] = biasGradient[j] + gradients.biasGradient[j];
        }

        for (int j = 1; j < netSizes.size() - 1; j++)
        {
            sw::Matrix<float> &weightMatrix = net.getWeights()[j];
            sw::Vector<float> &biasVector = net.getBiases()[j];

            weightMatrix = weightMatrix + weightGradient[j];
            biasVector = biasVector + biasGradient[j];
        }
    }

    // auto end = std::chrono::steady_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // printf("One minibatch took: %lld us\n", duration.count());
}

Trainer::ResultPair Trainer::feedforward(Net &net, sw::VectorView<float> &image)
{
    std::vector<uint16_t> layerSizes = net.getSizes();
    auto &weights = net.getWeights();
    auto &biases = net.getBiases();

    std::vector<sw::Vector<float>> activations;
    std::vector<sw::Vector<float>> zs;

    activations.push_back(sw::Vector(image.getStdVector()));

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

    size_t nInBetweenLayers = net.getSizes().size() - 1;

    std::vector<sw::Matrix<float>> weightGradient(nInBetweenLayers);
    std::vector<sw::Vector<float>> biasGradient(nInBetweenLayers);

    // example for: (net: (784, 30, 10), layerIndexes: (0, 1, 2). nLayers = 3).

    sw::Vector<float> sigmoidPrime;

    // Wanted change of z of the last layer resultPair.zs and .activations are both of length nLayers - 1 because it excludes the input layer
    //(so the last index is nLayers - 2).  sigmoid_prime(z) is the rate of change of the output, multiply this with your error
    //(wanted change in the output to get rid of this error) to calculate how much to change z. So for the last layer deltaZ has length 10.
    // scale deltaZ here to avoid a matrix-scalar product
    sw::Vector<float> deltaZ = errorVector.point_mult(net.sigmoid_prime(resultPair.zs[nInBetweenLayers - 1])) * (learningRate / miniBatchSize);
    weightGradient[nInBetweenLayers - 1] = deltaZ.outer(resultPair.activations[nInBetweenLayers - 1]);
    biasGradient[nInBetweenLayers - 1] = deltaZ;

    for (int i = 1; i < nInBetweenLayers; ++i)
    {
        sigmoidPrime = net.sigmoid_prime(resultPair.zs[nInBetweenLayers - i - 1]);
        // auto x = weights[nInBetweenLayers - i] * deltaZ;

        deltaZ = sigmoidPrime.point_mult(weights[nInBetweenLayers - i].transposeMult(deltaZ));

        weightGradient[nInBetweenLayers - i - 1] = deltaZ.outer(resultPair.activations[nInBetweenLayers - i - 1]);
        biasGradient[nInBetweenLayers - i - 1] = deltaZ;
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
