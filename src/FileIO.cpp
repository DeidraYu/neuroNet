#include <iostream>
#include <fstream>
#include <typeindex>
#include <variant>
#include <cassert>

#include "FileIO.hpp"

using namespace sw::math;

/*// Helper struct to store type information for serialization
struct TypeInfo
{
    std::type_index index;
    size_t size;
};

template <typename T>
TypeInfo getTypeInfo()
{
    return {std::type_index(typeid(T)), sizeof(T)};
}*/

// Function to save an std::vector<float> to a binary file
template <typename T>
void saveVectorToFile(const Vector<T> &vec, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        // Write the number of elements in the vector
        size_t numElements = vec.size();
        file.write(reinterpret_cast<const char *>(&numElements), sizeof(size_t));

        // Write the type information
        // TypeInfo typeInfo = getTypeInfo<T>();
        // file.write(reinterpret_cast<const char *>(&typeInfo), sizeof(TypeInfo));

        // Write the vector vec to the file
        file.write(reinterpret_cast<const char *>(vec.data()), numElements * sizeof(T));

        file.close();
        std::cout << "Vector saved to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Error: Unable to create file for writing." << std::endl;
    }
}

// Function to load an std::vector<float> from a binary file
std::vector<float> loadVectorFromFile(const std::string &filename)
{
    std::vector<float> vec;

    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        // Read the number of elements in the vector
        size_t numElements;
        file.read(reinterpret_cast<char *>(&numElements), sizeof(size_t));

        // Resize the vector and read the vec from the file
        vec.resize(numElements);
        file.read(reinterpret_cast<char *>(vec.data()), numElements * sizeof(float));

        file.close();
        std::cout << "Vector loaded from " << filename << std::endl;
    }
    else
    {
        std::cerr << "Error: Unable to open file for reading." << std::endl;
    }

    return vec;
}

/*void FileIO::saveNetwork(Network &network, std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        uint16_t numLayers = network.getNumLayers();
        file.write(reinterpret_cast<const char *>(&numLayers), sizeof(uint16_t));

        for (Layer &layer : network.getLayers())
        {
            uint16_t numInputs = layer.getNumInputs();
            uint16_t numOutputs = layer.getNumOutputs();
            file.write(reinterpret_cast<const char *>(&numInputs), sizeof(uint16_t));
            file.write(reinterpret_cast<const char *>(&numOutputs), sizeof(uint16_t));
            file.write(reinterpret_cast<const char *>(layer.getWeights()), layer.getWeights().size() * sizeof(float));
            file.write(reinterpret_cast<const char *>(layer.getBiases()), layer.getBiases().size() * sizeof(float));
        }

        file.close();
        std::cout << "Network saved to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }
}

Network FileIO::loadNetwork(const std::string &filename)
{
    Network network;

    std::ifstream file(filename, std::ios::binary);
    if (file.is_open())
    {
        uint16_t numLayers;
        file.read(reinterpret_cast<char *>(&numLayers), sizeof(uint16_t));

        for (uint16_t i = 0; i < numLayers; ++i)
        {
            uint16_t numInputs, numOutputs;
            file.read(reinterpret_cast<char *>(&numInputs), sizeof(uint16_t));
            file.read(reinterpret_cast<char *>(&numOutputs), sizeof(uint16_t));

            // Load weights into sw::Matrix<float>
            Matrix<float> weights(numOutputs, numInputs);
            file.read(reinterpret_cast<char *>(weights.vec()), weights.size() * sizeof(float));

            // Load biases into sw::Vector<float>
            Vector<float> biases(numOutputs);
            file.read(reinterpret_cast<char *>(biases.vec()), biases.size() * sizeof(float));

            network.addLayer(Layer(numInputs, numOutputs, weights, biases));
        }

        file.close();
        std::cout << "Network loaded from " << filename << std::endl;
    }
    else
    {
        std::cerr << "Error: Unable to open file for reading." << std::endl;
    }

    return network;
}*/