#include <chrono>

#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/Network.hpp"
#include "../src/Evaluator.hpp"
#include "../src/Trainer.hpp"

using namespace sw;

/*TEST(NetTest, trainNet)
{
    uint16_t nEpochs = 100;
    uint16_t miniBatchSize = 1;
    float learningRate = 10;

    std::vector<std::vector<uint8_t>> dataIn = {{0, 0}, {255, 0}, {0, 255}, {255, 255}};
    std::vector<uint8_t> labelsIn = {0, 1, 1, 0};
    std::vector<uint16_t> layersizes = {2, 2, 2};

    Net net(layersizes);
    net.setWeightsBiases(0.5f);

    Evaluator evaluator1(dataIn, labelsIn);
    Evaluator evaluator2(dataIn, labelsIn);
    Evaluator evaluator3(dataIn, labelsIn);
    Trainer trainer(dataIn, labelsIn);

    std::string bias1 = net.getBiases()[1].toString();
    std::string weight1 = net.getWeights()[1].toString();
    int score1 = evaluator2.evaluate(net);

    trainer.train(net, nEpochs, miniBatchSize, learningRate, evaluator1);

    std::string bias2 = net.getBiases()[1].toString();
    std::string weight2 = net.getWeights()[1].toString();
    int score2 = evaluator3.evaluate(net);

    printf("%s \n", bias1.c_str());
    printf("%s \n", bias2.c_str());
    printf("%s \n", weight1.c_str());
    printf("%s \n", weight2.c_str());
    printf("%d \n", score1);
    printf("%d \n", score2);

    EXPECT_NE(score1, score2);
}*/
