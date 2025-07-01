#pragma once
#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include "GLearn/NeuralNetwork/Model.hpp"
#include <cmath>
#include <vector>

namespace GLearn {
namespace NeuralNetwork {
class Network {
public:
  Network();
  Network(Network &&) = delete;
  Network(const Network &) = delete;
  Network &operator=(Network &&) = delete;
  Network &operator=(const Network &) = delete;
  ~Network();

  [[nodiscard]]
  static double_t CalculateNeuron(
      const std::vector<double_t> &_inputs,
      const std::vector<double_t> &_weights, double_t _bias,
      GLearn::NeuralNetwork::ACTIVATION_FUNCTION _func = Activation::None);

  [[nodiscard]]
  std::vector<double_t> Calculate(const std::vector<double_t> &_inputs,
                                  const Model &_model);

  Model Learn(const std::vector<std::vector<double_t>> &_inputs,
              const std::vector<std::vector<double_t>> &_expectedOutputs,
              const Model &_model, const double_t _learningRate = 0.001);
};
} // namespace NeuralNetwork
} // namespace GLearn
