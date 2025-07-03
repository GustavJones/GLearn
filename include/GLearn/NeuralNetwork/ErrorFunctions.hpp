#pragma once
#include <cmath>

typedef double_t (*ERROR_FUNCTION)(double_t _x, double_t _expected);

namespace GLearn {
namespace NeuralNetwork {
namespace Error {
double_t SquaredError(double_t _x, double_t _expected);

double_t AbsoluteError(double_t _x, double_t _expected);
}
} // namespace NeuralNetwork
} // namespace GLearn
