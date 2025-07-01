#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include <cmath>

namespace GLearn {
namespace NeuralNetwork {
namespace Activation {
double_t None(double_t _x, bool _derivative) {
  if (_derivative) {
    return 1;
  } else {
    return _x;
  }
}

double_t Sigmoid(double_t _x, bool _derivative) {
  if (_derivative) {
    double_t recurse = GLearn::NeuralNetwork::Activation::Sigmoid(_x);
    return recurse * (1 - recurse);
  } else {
    return 1 / (1 + std::exp(-_x));
  }
}
} // namespace Activation
} // namespace NeuralNetwork
} // namespace GLearn
