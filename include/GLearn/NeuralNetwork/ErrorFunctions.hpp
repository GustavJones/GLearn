#pragma once
#include <cmath>
#include <string>
#include <map>

typedef double_t (*ERROR_FUNCTION)(double_t _x, double_t _expected, bool _derivative);

namespace GLearn {
namespace NeuralNetwork {
namespace Error {
double_t SquaredError(double_t _x, double_t _expected, bool _derivative = false);

double_t AbsoluteError(double_t _x, double_t _expected, bool _derivative = false);

static const std::map<std::string, ERROR_FUNCTION> Functions = { {"SQUARED_ERROR", SquaredError}, {"ABSOLUTE_ERROR", AbsoluteError} };
}
} // namespace NeuralNetwork
} // namespace GLearn
