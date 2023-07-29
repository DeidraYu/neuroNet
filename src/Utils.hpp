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
