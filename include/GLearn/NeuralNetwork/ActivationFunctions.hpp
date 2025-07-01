#pragma once
#include <cmath>

namespace GLearn {
namespace NeuralNetwork {
typedef double_t (*ACTIVATION_FUNCTION)(double_t, bool);
}
} // namespace GLearn

namespace GLearn {
namespace NeuralNetwork {
namespace Activation {
double_t None(double_t _x, bool _derivative = false);
double_t Sigmoid(double_t _x, bool _derivative = false);

} // namespace Activation
} // namespace NeuralNetwork
} // namespace GLearn
