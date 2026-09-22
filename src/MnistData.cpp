#include <chrono>

#include "MnistData.hpp"
#include "Utils.hpp"

void MnistData::processMNIST()
{
    processMNIST(0, 0); // no limits on the size of the training and test set
}

void MnistData::processMNIST(size_t trainingLimit, size_t testLimit)
{
    // Load MNIST data
    dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION, trainingLimit, testLimit);

    // Get the actual sizes of the training and testing data sets.
    numTrainImages = dataset.training_images.size();
    numTestImages = dataset.test_images.size();

    // Allocate the memory for the training/testing data/labels
    train_data = std::vector<Vector<float>>(numTrainImages);
    train_labels = std::vector<Vector<float>>(numTrainImages);
    test_data = std::vector<Vector<float>>(numTestImages);
    test_labels = std::vector<Vector<float>>(numTestImages);

    // Convert and scale the image data and create one-hot encoded labels, both for training and test sets
    for (uint32_t i = 0; i < numTrainImages; ++i)
    {
        train_data[i] = Vector<float>(Vector<uint8_t>(dataset.training_images[i])) * (1.0f / 256.0f);
        train_labels[i] = oneHotEncode(dataset.training_labels[i], 10);
    }

    for (uint32_t i = 0; i < numTestImages; ++i)
    {
        test_data[i] = Vector<float>(Vector<uint8_t>(dataset.test_images[i])) * (1.0f / 256.0f);
        test_labels[i] = oneHotEncode(dataset.test_labels[i], 10);
    }
}
