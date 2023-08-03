#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "../src/Layer.hpp"
#include "../src/Network.hpp"
#include "../src/FileIO.hpp"

#include "../src/math/Vector.hpp"

using namespace sw::math;

TEST(FileIoTest, initialize)
{
    Vector<float> testVec{1.0f, 2.0f, 3.0f, 5.0f};
    std::string fileName = "/home/koenr/projects/neuroNetSaves/testVec.bin";

    // saveVectorToFile(testVec, fileName);
}

TEST(FileIoTest, initialize2)
{
    Vector<float> testVec{1.0f, 2.0f, 3.0f, 5.0f};
    std::string fileName = "/home/koenr/projects/neuroNetSaves/testVec.bin";

    // saveVectorToFile(testVec, fileName);
}