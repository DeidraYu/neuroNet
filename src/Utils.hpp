#pragma once

#include <iostream>
#include <vector>

#include "Matrix.hpp"

void printNumber(std::vector<uint8_t> &vec);

void printProgress(int part, int total, std::string labelString);

void printSW(sw::Vector<float> &vec);
