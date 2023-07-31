#pragma once

#include <iostream>
#include <vector>

// #include "Matrix.hpp"
#include "math/Vector.hpp"
#include "mnist/mnist_reader.hpp"

using namespace sw::math;

void printNumber(std::vector<uint8_t> &vec);

void printProgress(int part, int total, std::string labelString);

void printSW(Vector<float> &vec);

void processMNIST();

Vector<float> oneHotEncode(int value, int numClasses);

namespace sw
{
    namespace prof
    {
        class Counter
        {
        public:
            Counter();
            Counter(float time_ns);
            void update(float time_ns);

            uint64_t count() const;
            float time_ns() const;
            float mean_ns() const;

        private:
            float m_time_ns;
            uint64_t m_count;
        };

        class Times
        {
        public:
            static void update(const std::string name, float time_ns);
            static const Counter &getCounter(const std::string name);
            static std::string toString(int nEpochs);

        private:
            static std::unordered_map<std::string, Counter> m_times;
        };

        class Measure
        {
        public:
            Measure(std::string name);
            ~Measure();

        private:
            std::string m_name;
            std::chrono::steady_clock::time_point m_start;
            std::chrono::steady_clock::time_point m_end;

            int64_t m_time_ns;
        };

    } // namespace prof
} // namespace sw
