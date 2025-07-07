#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include <cmath>

namespace GLearn {
namespace NeuralNetwork {
namespace Error {
double_t SquaredError(double_t _x, double_t _expected, bool _derivative) {
  if (!_derivative)
  {
    return std::pow(_x - _expected, 2);
  }
  else
  {
    return 2 * (_x - _expected);
  }
}

double_t AbsoluteError(double_t _x, double_t _expected, bool _derivative) {
  if (!_derivative)
  {
    return std::abs(_x - _expected);
  }
  else
  {
    // Might cause errors because abs functions does not have a specific gradient at it's turning point
    if (_x < _expected)
    {
      return -1;
    }
    else
    {
      return 1;
    }
  }
}
} // namespace Error
} // namespace NeuralNetwork
} // namespace GLearn
