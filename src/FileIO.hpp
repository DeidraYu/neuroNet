#pragma once

#include <iostream>
#include <fstream>
#include <typeindex>
#include <variant>
#include <cassert>
#include <vector>

#include "math/Vector.hpp"

using namespace sw::math;

// #include "Network.hpp"

/*class FileIO
{
public:
    static void saveNetwork(Network &network, std::string filename);
    static Network loadNetwork(const std::string &filename);
};

template <typename T>
void saveVectorToFile(const Vector<T> &vec, const std::string &filename);

std::vector<float> loadVectorFromFile(const std::string &filename);*/

template <typename T>
sw::math::Vector<T> parseBinaryData(std::vector<uint8_t> binaryData)
{
    auto itr = binaryData.begin();

    // Extract type information from binary data
    TypeInfo typeInfo{std::type_index(typeid(0)), 0};
    std::copy_n(itr, sizeof(TypeInfo), reinterpret_cast<uint8_t *>(&typeInfo));
    itr += sizeof(TypeInfo);

    // Extract the number of elements from binary data
    size_t numElements;
    std::copy_n(itr, sizeof(size_t), reinterpret_cast<uint8_t *>(&numElements));
    itr += sizeof(size_t);

    // Extract vector data and construct the Vector

    std::vector<T> dataVector(numElements);

    std::copy_n(itr, numElements * sizeof(T), reinterpret_cast<uint8_t *>(dataVector.data()));
    itr += numElements * sizeof(T);

    Vector<T> outputVector = Vector<T>(dataVector);

    return outputVector;
}