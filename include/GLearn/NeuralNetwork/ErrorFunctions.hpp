#pragma once
#include <cmath>

namespace GLearn {
namespace NeuralNetwork {
typedef double_t (*ERROR_FUNCTION)(double_t, double_t, bool);
}
} // namespace GLearn

namespace GLearn {
namespace NeuralNetwork {
namespace Error {
double_t SquaredError(double_t _x, double_t _expected,
                      bool _derivative = false);
}
} // namespace NeuralNetwork
} // namespace GLearn
