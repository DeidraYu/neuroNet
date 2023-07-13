#include <cstdio>

#include "Net.hpp"

Net::Net(std::vector<uint16_t> layerSizes) : m_layerSizes(layerSizes) {}

void Net::printInfo()
{
    printf("Number of nodes per layer:\n");
    for (size_t layerIndex = 0; layerIndex < m_layerSizes.size(); ++layerIndex)
    {
        printf("    %2lld: %d\n", layerIndex, m_layerSizes[layerIndex]);
    }
}
