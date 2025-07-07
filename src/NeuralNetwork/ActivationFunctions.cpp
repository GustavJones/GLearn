#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace GLearn {
namespace NeuralNetwork {
namespace Activation {
double_t None(double_t _x, bool _derivative) {
  if (!_derivative)
  {
    return _x;
  }
  else
  {
    return 1.0;
  }
}

double_t Sigmoid(double_t _x, bool _derivative) {
  double_t recurse;

  if (!_derivative)
  {
    return 1.0 / (1.0 + std::exp(-_x));
  }
  else
  {
    recurse = Sigmoid(_x, false);
    return recurse * (1.0 - recurse);
  }
}

double_t ReLu(double_t _x, bool _derivative) {
  if (!_derivative)
  {
    return std::max<double_t>(0.0, _x);
  }
  else
  {
    if (_x < 0)
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }
}

double_t HyperbolicTangent(double_t _x, bool _derivative) {
  if (!_derivative)
  {
    return (std::exp(_x) - std::exp(-_x)) / (std::exp(_x) + std::exp(-_x));
  }
  else
  {
    return 1 - std::pow(HyperbolicTangent(_x, false), 2);
  }
}

} // namespace Activation
} // namespace NeuralNetwork
} // namespace GLearn
