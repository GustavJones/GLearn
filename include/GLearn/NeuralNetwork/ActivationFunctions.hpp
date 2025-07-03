#pragma once
#include <cmath>

typedef double_t(*ACTIVATION_FUNCTION)(double_t _x);

namespace GLearn {
namespace NeuralNetwork {
namespace Activation {


double_t None(double_t _x);
double_t Sigmoid(double_t _x);
double_t ReLu(double_t _x);
double_t HyperbolicTangent(double_t _x);

} // namespace Activation
} // namespace NeuralNetwork
} // namespace GLearn
