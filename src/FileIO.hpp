#pragma once

#include "Network.hpp"

class FileIO
{
public:
    /*static void saveNetwork(Network &network, std::string filename);
    static Network loadNetwork(const std::string &filename);*/
};

template <typename T>
void saveVectorToFile(const Vector<T> &vec, const std::string &filename);

std::vector<float> loadVectorFromFile(const std::string &filename);