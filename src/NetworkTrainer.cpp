#include "NetworkTrainer.hpp"

void NetworkTrainer::trainNet(Network &network, uint32_t nEpochs, float learningRate, uint16_t miniBatchSize, MnistData &mnistData)
{
    char progressLabel[16];
    for (uint32_t i = 0; i < nEpochs; ++i)
    {
        snprintf(progressLabel, 16, "%d / %d", i + 1, nEpochs);
        network.progressLabel = progressLabel;
        network.trainMiniBatches(miniBatchSize, learningRate, mnistData.train_data, mnistData.train_labels);
    }
    printf("\n"); // because the progress bar has no newline.
}

uint32_t NetworkTrainer::evalNet(Network &network, MnistData &mnistData)
{
    uint32_t score = 0;

    for (uint32_t i = 0; i < mnistData.numTestImages; ++i)
    {
        network.feedforward(mnistData.test_data[i]);

        if (network.getOutput().argmax() == mnistData.dataset.test_labels[i])
        {
            ++score;
        }
        // printf("%d: ", mnistData.dataset.test_labels[i]);
        // std::cout << network.getOutput().toString() << std::endl;
    }

    return score;
}