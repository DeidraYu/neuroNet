#include "NetworkTrainer.hpp"

void NetworkTrainer::trainNet(Network &network, uint32_t nEpochs, float learningRate, uint16_t miniBatchSize, MnistData &mnistData)
{
    sw::prof::Measure M("NetworkTrainer.trainNet");

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
    sw::prof::Measure M("NetworkTrainer.evalNet");

    uint32_t score = 0;

    for (uint32_t i = 0; i < mnistData.numTestImages; ++i)
    {
        network.fastFeedforward(mnistData.test_data[i]);

        Vector<float> output = network.getOutput();
        if (output.argmax() == mnistData.dataset.test_labels[i])
        {
            ++score;
        }
        // else
        // {
        //     int dummy = 0;
        //     Vector<uint8_t> intData = mnistData.test_data[i] * 256;
        //     std::vector<uint8_t> stdVector(intData.size());
        //     stdVector.assign(intData.data(), intData.data() + intData.size());
        //     printNumber(stdVector);
        //     std::cout << "label: " << uint16_t(mnistData.dataset.test_labels[i]) << std::endl;
        //     std::cout << "inferred: " << output.argmax() << std::endl;
        //     std::cout << std::endl;
        // }
        // printf("%d: ", mnistData.dataset.test_labels[i]);
        // std::cout << network.getOutput().toString() << std::endl;
    }

    return score;
}