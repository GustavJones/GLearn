#pragma once
#include <cmath>
#include <vector>
#include <cstdint>

namespace GLearn {
namespace NeuralNetwork {
double_t Mean(const std::vector<double_t> &_values, bool _derivative = false);
}
} // namespace GLearn
