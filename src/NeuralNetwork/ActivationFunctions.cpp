#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace GLearn {
namespace NeuralNetwork {
namespace Activation {
double_t None(double_t _x) {
  return _x;
}

double_t Sigmoid(double_t _x) {
  return 1.0 / (1.0 + std::exp(-_x));

}

double_t ReLu(double_t _x) {
  return std::max<double_t>(0, _x);
}

double_t HyperbolicTangent(double_t _x) {
  return (std::exp(_x) - std::exp(-_x)) / (std::exp(_x) + std::exp(-_x));
}

} // namespace Activation
} // namespace NeuralNetwork
} // namespace GLearn
