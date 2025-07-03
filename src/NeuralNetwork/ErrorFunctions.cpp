#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include <cmath>

namespace GLearn {
namespace NeuralNetwork {
namespace Error {
double_t SquaredError(double_t _x, double_t _expected) {
  return std::pow(_expected - _x, 2);
}

double_t AbsoluteError(double_t _x, double_t _expected) {
  return std::sqrt(SquaredError(_x, _expected));
}
} // namespace Error
} // namespace NeuralNetwork
} // namespace GLearn
