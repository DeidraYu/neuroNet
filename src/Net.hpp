#pragma once

#include <vector>

class Net
{
public:
    Net(std::vector<uint16_t> layerSizes);

    void printInfo();

private:
    std::vector<uint16_t> m_layerSizes;
};