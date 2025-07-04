#pragma once
#include <cmath>
#include <string>
#include <map>

typedef double_t(*ACTIVATION_FUNCTION)(double_t _x);

namespace GLearn {
namespace NeuralNetwork {
namespace Activation {


double_t None(double_t _x);
double_t Sigmoid(double_t _x);
double_t ReLu(double_t _x);
double_t HyperbolicTangent(double_t _x);

static const std::map<std::string, ACTIVATION_FUNCTION> Functions = { {"NONE", None}, {"SIGMOID", Sigmoid}, {"RELU", ReLu}, {"HYPERBOLIC_TANGENT", HyperbolicTangent} };
} // namespace Activation
} // namespace NeuralNetwork
} // namespace GLearn
