#include <chrono>

#include "MnistData.hpp"
#include "Utils.hpp"

void MnistData::processMNIST()
{
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    auto start1 = std::chrono::steady_clock::now();

    // Load MNIST data
    dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    auto end1 = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    std::cout << "Time reading mnist dataset from disk: " << duration1.count() << " milliseconds" << std::endl;

    numTrainImages = dataset.training_images.size();
    numTestImages = dataset.test_images.size();

    train_data = std::vector<sw::Vector<float>>(numTrainImages);
    train_labels = std::vector<sw::Vector<float>>(numTrainImages);

    test_data = std::vector<sw::Vector<float>>(numTestImages);
    test_labels = std::vector<sw::Vector<float>>(numTestImages);

    {
        for (uint32_t i = 0; i < numTrainImages; ++i)
        {
            sw::VectorView train_image_int(&(dataset.training_images[i]));
            sw::Vector<float> train_image = train_image_int * (1.0f / 256.0f);

            train_data[i] = std::move(train_image);

            train_labels[i] = oneHotEncode(dataset.training_labels[i], 10);
        }

        for (uint32_t i = 0; i < numTestImages; ++i)
        {
            sw::VectorView test_image_int(&(dataset.test_images[i]));
            sw::Vector<float> test_image = test_image_int * (1.0f / 256.0f);

            test_data[i] = std::move(test_image);

            test_labels[i] = oneHotEncode(dataset.test_labels[i], 10);
        }
    }
}
