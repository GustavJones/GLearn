#include "GLearn/NeuralNetwork/Network.hpp"
#include "GLearn/NeuralNetwork/Model.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace GLearn {
namespace NeuralNetwork {
Network::Network() {}

Network::~Network() {}

double_t
Network::CalculateNeuron(const std::vector<double_t> &_inputs,
                         const std::vector<double_t> &_weights, double_t _bias,
                         GLearn::NeuralNetwork::ACTIVATION_FUNCTION _func) {
  double_t out = 0;

  for (size_t i = 0; i < _inputs.size(); i++) {
    out += _inputs[i] * _weights[i];
  }

  out += _bias;
  return _func(out, false);
}

std::vector<double_t> Network::Calculate(const std::vector<double_t> &_inputs,
                                         const Model &_model) {
  double_t temp;
  std::vector<double_t> layerInputs = _inputs;
  std::vector<double_t> layerOutputs;

  if (!_model.IsValid()) {
    throw std::runtime_error("Invalid model weights and biases");
  }

  for (size_t i = 0; i < _model.weights[0].size(); i++) {
    if (_model.weights[0][i].size() != _inputs.size()) {
      throw std::runtime_error("Invalid input weight combinations");
    }
  }

  // For every layer
  for (size_t i = 0; i < _model.weights.size(); i++) {
    // For every neuron in layer
    for (size_t j = 0; j < _model.weights[i].size(); j++) {
      layerOutputs.push_back(CalculateNeuron(layerInputs, _model.weights[i][j],
                                             _model.biases[i][j]));
    }

    layerInputs = layerOutputs;
    layerOutputs.clear();
  }

  return layerInputs;
}

Model Network::Learn(const std::vector<std::vector<double_t>> &_inputs,
                     const std::vector<std::vector<double_t>> &_expectedOutputs,
                     const Model &_model, const double_t _learningRate) {
  Model out;
  std::vector<double_t> datapoint = _inputs[0];
  std::vector<double_t> datapointOutput,
      datapointExpectedOutput = _expectedOutputs[0];

  datapointOutput = Calculate(datapoint, _model);

  return _model;
}
} // namespace NeuralNetwork
} // namespace GLearn
