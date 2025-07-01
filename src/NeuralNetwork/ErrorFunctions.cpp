#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include <cmath>

namespace GLearn {
namespace NeuralNetwork {
namespace Error {
double_t SquaredError(double_t _x, double_t _expected, bool _derivative) {
  if (_derivative) {
    return -2 * std::pow(_expected - _x, 2);
  } else {
    return std::pow(_expected - _x, 2);
  }
}
} // namespace Error
} // namespace NeuralNetwork
} // namespace GLearn
