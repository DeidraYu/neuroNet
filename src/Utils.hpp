#pragma once

#include <iostream>
#include <vector>

#include "Matrix.hpp"
#include "mnist/mnist_reader.hpp"

void printNumber(std::vector<uint8_t> &vec);

int getConsoleWidth();

void printProgress(int part, int total, std::string labelString);

void printSW(sw::Vector<float> &vec);

void processMNIST();

sw::Vector<float> oneHotEncode(int value, int numClasses);
