#include "Net.hpp"

Net::Net(std::vector<uint16_t> layerSizes) : m_layerSizes(layerSizes) {}

void Net::printInfo()
{
    printf("Number of nodes per layer:\n");
    for (int layerIndex = 0; layerIndex < m_layerSizes.size(); ++layerIndex)
    {
        printf("    %2d: %d\n", layerIndex, m_layerSizes[layerIndex]);
    }
}
