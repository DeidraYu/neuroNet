#include <iostream>
#include <vector>
#include <chrono>

#include <random>
#include <algorithm>
#include <ranges>

#include "mnist/mnist_reader.hpp"
#include "Net.hpp"
#include "Trainer.hpp"
#include "MathUtils.hpp"

void doit()
{
    printf("Hello World!\n");
}

void printNumber(std::vector<uint8_t> &vec)
{
    for (int y = 0; y < 28; y += 2)
    {
        for (int x = 0; x < 28; ++x)
        {
            // uint8_t pixelValue = dataset.training_images[++index];
            uint8_t pix_top = vec[x + 28 * y];
            uint8_t pix_bot = vec[x + 28 * (y + 1)];

            if ((pix_top < 128) && (pix_bot < 128))
            {
                printf(" ");
            }

            if ((pix_top < 128) && (pix_bot >= 128))
            {
                printf("%c", 0xDC);
            }

            if ((pix_top >= 128) && (pix_bot < 128))
            {
                printf("%c", 0xDF);
            }

            if ((pix_top >= 128) && (pix_bot >= 128))
            {
                printf("%c", 0xDB);
            }
        }
        printf("\n");
    }
}

void shuffleTest()
{
    std::vector<char> u = {'A', 'B', 'C', 'D', 'E'};
    std::vector<char> v = {'a', 'b', 'c', 'd', 'e'};

    // std::vector<std::pair> myPairs;

    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // auto genCopy = gen;

    // Shuffle the entries in u using the same shuffle for v
    std::ranges::shuffle(u, std::mt19937(gen));
    std::ranges::shuffle(v, gen);

    std::ranges::shuffle(u, std::mt19937(gen));
    std::ranges::shuffle(v, gen);

    // Print the shuffled u and v vectors
    for (const auto &val : u)
    {
        std::cout << val << ' ';
    }
    std::cout << std::endl;

    for (const auto &val : v)
    {
        std::cout << val << ' ';
    }
    std::cout << std::endl;
}

/*
class VecView
{
public:
    VecView(std::vector<int> &vec) : m_vec{vec} {}
    void print()
    {
        printf("vec[2]: %d\n", m_vec[2]);
    }
    void set()
    {
        m_vec[2] = 9;
    }

    int operator[](std::vector<int>::size_type index) { return m_vec[index]; }
    const int &operator[](std::vector<int>::size_type index) const { return m_vec[index]; }

private:
    std::vector<int> &m_vec;
};

class Vec : public VecView
{
public:
    Vec(std::vector<int> vec) : m_vecStorage{vec}, VecView{m_vecStorage} {}
    // Vec(std::vector<int> &&vec) : m_vecStorage{vec}, VecView{m_vecStorage} {}

private:
    std::vector<int> m_vecStorage;
};
*/

void vectorTest()
{
    /*
    std::vector<int> vec{0, 1, 2};
    VecView vw{vec};
    vw.set();

    Vec v1{vec}; // works
    v1.print();
    printf("%d\n", v1[2]);

    Vec v2(vec);
    v2.print();
    printf("%d\n", v2[2]);

    Vec v3(std::move(vec));
    v3.print();
    printf("%d\n", v3[2]);
    exit(0);
    */
    std::vector<float> x{1.0f, 2.0f, 3.0f};
    // std::vector<int> y{2, 3, 4};

    sw::Vector<float> u{x};
    // sw::Vector v(y);

    // sw::Vector u{1.0f, 2.0f, 3.0f};

    // exit(0);

    sw::Vector v{2, 3, 4};

    printf("u[1]: %f\n", u[1]);

    /*
    v[2] = 5;
    printf("v[2]: %d\n", v[2]);

    float prod = u * v;
    printf("prod: %f\n", prod);

    printf("scalarProd: %f\n", 0.5f * u * v);
    printf("scalarProd: %f\n", (0.5f * u) * v);
    printf("scalarProd: %f\n", 0.5f * (u * v));

    printf("u = %s\n", u.toString().c_str());
    printf("v = %s\n", v.toString().c_str());

    u.point_mult(v);
    u.point_mult(u);

    // sw::Vector pwProd = u.point_mult(v);
    auto pwProd = u.point_mult(v);

    pwProd.point_mult(v);

    pwProd.point_mult(u);
    */
    // printf("pwProd.point_mult(u) = %s\n", pwProd.point_mult(u).toString().c_str());

    // sw::Vector pwProdProd = pwProd.point_mult(u);

    // printf("pwProd = %s\n", pwProd.toString().c_str());
    // printf("scalarProd = %s\n", u.point_mult(v).toString().c_str());

    // printf("scalarProd = %s\n", (0.5f * u.point_mult(v)).toString().c_str());

    // sw::Vector w{1.0f, 2.0f, 3.0f};

    // sw::Vector c{w};

    // printf("w = %s\n", w.toString().c_str());
    // printf("c = %s\n", c.toString().c_str());

    // w[0] = 0.0f;
    // c[0] = 10.0f;

    // printf("w = %s\n", w.toString().c_str());
    // printf("c = %s\n", c.toString().c_str());
}

int main()
{

    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    auto start1 = std::chrono::steady_clock::now();

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    auto end1 = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    std::cout << "Time reading mnist dataset from disk: " << duration1.count() << " milliseconds" << std::endl;

    if (false)
    {
        printNumber(dataset.training_images[0]);
        printNumber(dataset.training_images[1]);
        printNumber(dataset.training_images[2]);
    }

    Net net({28 * 28, 30, 10});
    // net.printInfo();

    Trainer trainer(dataset.training_images, dataset.training_labels);

    {
        auto start2 = std::chrono::steady_clock::now();

        int16_t nEpochs = 10;
        uint16_t miniBatchSize = 5;
        float learningRate = 2.0f;

        trainer.train(nEpochs, miniBatchSize, learningRate);

        auto end2 = std::chrono::steady_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
        printf("Time permuting data set %d times: %lld milliseconds\n", nEpochs, duration2.count());
    }

    // shuffleTest();
    vectorTest();

    return 0;
}
