#include <chrono>
#include <random>

#include "gtest/gtest.h"

#include "../src/Matrix.hpp"

using namespace sw;

template <typename T>
class MatrixPerformance
{
public:
    MatrixPerformance(int size, int numRepeats) : A(size, size), x(size), numRepeats(numRepeats)
    {

        if constexpr (std::is_integral_v<T>)
        {
            A = Matrix<T>::rand(size, size, 0, std::numeric_limits<T>::max());
            x = Vector<T>::rand(size, 0, std::numeric_limits<T>::max());
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            A = Matrix<T>::rand(size, size, 0.0f, 0.1f);
            x = Vector<T>::rand(size, 0.0f, 0.1f);
        }
    }

    float run()
    {
        auto start = std::chrono::steady_clock::now();

        for (int n = 0; n < numRepeats; n++)
        {
            y = A * x;
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        long long time_us = duration.count();
        float totalTime_ms = time_us / 1'000.0f;
        float meanTime_ms = totalTime_ms / numRepeats;
        return meanTime_ms;
    }

    Matrix<T> A;
    Vector<T> x;
    Vector<T> y;
    int numRepeats;
};

template <typename T>
struct MatrixRunner
{

    static void run(std::string typeString, int size, int numRepeats)
    {
        MatrixPerformance<T> matrixPerformance(size, numRepeats);
        printf(".");
        float meanTime_ms = matrixPerformance.run();
        printf("Type: %8s  Size: %5d  Time: %7.2f ms\n", typeString.c_str(), size, meanTime_ms);

        // return meanTime_ms;
    }
};

TEST(Numbers, numbers)
{
    {
        uint64_t a = std::numeric_limits<uint64_t>::max();
        unsigned long long b = std::numeric_limits<unsigned long long>::max();
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << a << std::endl
                  << std::endl;
        std::uniform_int_distribution<uint64_t> dis(0, a);
    }
}

TEST(MatrixPerformance, multiply_with_vector)
{

#ifndef NDEBUG
    std::cout << std::endl;
    std::cout << "******************************************" << std::endl;
    std::cout << "* Waring! You are running a Debug build. *" << std::endl;
    std::cout << "******************************************" << std::endl;
    std::cout << std::endl;
#endif

    //                           type    size  numRepeat
    MatrixRunner<uint8_t>::run("uint8_t", 100, 10000);
    MatrixRunner<uint8_t>::run("uint8_t", 1000, 100);
    MatrixRunner<uint8_t>::run("uint8_t", 10000, 1);
    printf("\n");

    MatrixRunner<uint16_t>::run("uint16_t", 100, 10000);
    MatrixRunner<uint16_t>::run("uint16_t", 1000, 100);
    MatrixRunner<uint16_t>::run("uint16_t", 10000, 1);
    printf("\n");

    MatrixRunner<uint32_t>::run("uint32_t", 100, 10000);
    MatrixRunner<uint32_t>::run("uint32_t", 1000, 100);
    MatrixRunner<uint32_t>::run("uint32_t", 10000, 1);
    printf("\n");

    MatrixRunner<uint64_t>::run("uint64_t", 100, 10000);
    MatrixRunner<uint64_t>::run("uint64_t", 1000, 100);
    MatrixRunner<uint64_t>::run("uint64_t", 10000, 1);
    printf("\n");

    MatrixRunner<float>::run("float", 100, 10000);
    MatrixRunner<float>::run("float", 1000, 100);
    MatrixRunner<float>::run("float", 10000, 1);
    printf("\n");

    MatrixRunner<double>::run("double", 100, 10000);
    MatrixRunner<double>::run("double", 1000, 100);
    MatrixRunner<double>::run("double", 10000, 1);
    printf("\n");
}
