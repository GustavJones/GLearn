#pragma once
#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include "GLearn/NeuralNetwork/Model.hpp"
#include "GLearn/NeuralNetwork/Derivatives.hpp"
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

private:
  [[nodiscard]]
  double_t
  _CalculateNeuron(const std::vector<double_t> &_inputs,
                   const std::vector<double_t> &_weights, double_t _bias);

  [[nodiscard]]
  Model
  _TrainIteration(const std::vector<std::vector<double_t>> &_inputs,
                  const std::vector<std::vector<double_t>> &_expectedOutputs,
                  Model _model, const double_t _learningRate);

  [[nodiscard]]
  Model
  _ModifyParameters(Model _model,
                    const std::vector<std::vector<double_t>> &_inputs,
                    const std::vector<std::vector<double_t>> &_expectedOutputs, const double_t _learningRate);

  [[nodiscard]]
  Model
  _ModifyWeight(Model _model, const size_t _layer, const size_t _neuron,
                const size_t _weight,
                const std::vector<std::vector<double_t>> &_inputs,
                const std::vector<std::vector<double_t>> &_expectedOutputs, 
                const double_t _learningRate);

  [[nodiscard]]
  Model _ModifyBias(Model _model, const size_t _layer, const size_t _neuron,
                    const std::vector<std::vector<double_t>> &_inputs,
                    const std::vector<std::vector<double_t>> &_expectedOutputs,
                    const double_t _learningRate);

  [[nodiscard]]
  double_t _CalculateDatapointError(const Model& _model, const std::vector<double_t>& _input, const std::vector<double_t>& _expectedOutput);

  [[nodiscard]]
  Derivatives _CalculateDeltas(const Model &_model, const std::vector<double_t>& _input, const std::vector<double_t>& _expectedOutput);

  [[nodiscard]]
  double_t _CalculateBiasCostDerivative(const size_t _layer, const size_t _neuron, const std::vector<double_t>& _input,
    const std::vector<double_t>& _expectedOutput,
    const Model& _model);

  [[nodiscard]]
  double_t _CalculateWeightCostDerivative(const size_t _layer, const size_t _neuron, const size_t _weight, const std::vector<double_t>& _input,
    const std::vector<double_t>& _expectedOutput,
    const Model& _model);

  [[nodiscard]]
  double_t _CalculateParentDerivative(const size_t _layer, const size_t _neuron, const Model& _model, const std::vector<std::vector<double_t>>& _activatedStructure, const std::vector<std::vector<double_t>>& _unactivatedStructure, const std::vector<double_t>& _expectedOutput);

public:
  [[nodiscard]]
  double_t
  CalculateMeanError(const std::vector<std::vector<double_t>> &_inputs,
                     const std::vector<std::vector<double_t>> &_expectedOutputs,
                     const Model &_model);

  [[nodiscard]]
  std::vector<double_t> CalculateOutput(const std::vector<double_t> &_inputs,
                                        const Model &_model);

  [[nodiscard]]
  std::vector<std::vector<double_t>>
  CalculateStructure(const std::vector<double_t> &_inputs, const Model &_model);

  [[nodiscard]]
  std::vector<std::vector<double_t>>
  CalculateUnactivatedStructure(const std::vector<double_t>& _inputs, const Model& _model);

  [[nodiscard]]
  Model Learn(const std::vector<std::vector<double_t>> &_inputs,
              const std::vector<std::vector<double_t>> &_expectedOutputs,
              Model _model, const size_t _iterations,
              double_t _learningRate = 0.1, const double_t _variableLearningReduceRate = 0.1, const size_t _logInterval = 100);
};
} // namespace NeuralNetwork
} // namespace GLearn
