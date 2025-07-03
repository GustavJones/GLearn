#include "GLearn/NeuralNetwork/Network.hpp"
#include "GLearn/NeuralNetwork/Mean.hpp"
#include "GLearn/NeuralNetwork/Model.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <thread>
#include <future>

static const double_t SLOPE_SIZE = 0.000001;

namespace GLearn {
namespace NeuralNetwork {
Network::Network() {}

Network::~Network() {}

double_t
Network::_CalculateNeuron(const std::vector<double_t> &_inputs,
                          const std::vector<double_t> &_weights, double_t _bias) {
  double_t out = 0;

  for (size_t i = 0; i < _inputs.size(); i++) {
    out += _inputs[i] * _weights[i];
  }

  out += _bias;
  return out;
}

Model Network::_TrainIteration(
    const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs, Model _model,
    const double_t _learningRate) {
  if (_inputs.size() != _expectedOutputs.size()) {
    throw std::runtime_error("Input output pairs not the same amount");
  }

  _model = _ModifyParameters(_model, _inputs, _expectedOutputs, _learningRate);

  return _model;
}

Model Network::_ModifyParameters(
    Model _model, const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs, const double_t _learningRate) {

  for (size_t layer = 0; layer < _model.weights.size(); layer++) {
    for (size_t neuron = 0; neuron < _model.weights[layer].size(); neuron++)
    {
      _model = _ModifyBias(_model, layer, neuron, _inputs, _expectedOutputs, _learningRate);

      for (size_t weight = 0; weight < _model.weights[layer][neuron].size(); weight++)
      {
        _model = _ModifyWeight(_model, layer, neuron, weight, _inputs, _expectedOutputs, _learningRate);
      }
    }
  }

  return _model;
}

Model Network::_ModifyWeight(
    Model _model, const size_t _layer, const size_t _neuron,
    const size_t _weight, const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs, const double_t _learningRate) {
  Model newModel;

  double_t slope; // dError / dWeight
  double_t dError, dWeight;

  double_t finalMeanError, startMeanError;

  dWeight = SLOPE_SIZE;

  newModel = _model;
  newModel.weights[_layer][_neuron][_weight] += dWeight;

  finalMeanError = CalculateMeanError(_inputs, _expectedOutputs, newModel);
  startMeanError = CalculateMeanError(_inputs, _expectedOutputs, _model);

  dError = finalMeanError - startMeanError;;

  slope = dError / dWeight;

  _model.weights[_layer][_neuron][_weight] -= slope * _learningRate;
  return _model;
}

Model Network::_ModifyBias(
    Model _model, const size_t _layer, const size_t _neuron, 
    const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs, 
    const double_t _learningRate) {
  Model newModel;

  double_t slope; // dError / dWeight
  double_t dError, dBias;

  double_t finalMeanError, startMeanError;

  dBias = SLOPE_SIZE;

  newModel = _model;
  newModel.biases[_layer][_neuron] += dBias;

  auto finalThread = std::async(std::launch::async, &Network::CalculateMeanError, this, _inputs, _expectedOutputs, newModel);
  auto startThread = std::async(std::launch::async, &Network::CalculateMeanError, this, _inputs, _expectedOutputs, _model);

  finalThread.wait();
  startThread.wait();

  finalMeanError = finalThread.get();
  startMeanError = startThread.get();

  dError = finalMeanError - startMeanError;

  slope = dError / dBias;

  _model.biases[_layer][_neuron] -= slope * _learningRate;
  return _model;
}

double_t Network::_CalculateDatapointError(const Model &_model, const std::vector<double_t> &_input, const std::vector<double_t> &_expectedOutput) {
  double_t error = 0;

  std::vector<double_t> datapointOutput;
  datapointOutput = CalculateOutput(_input, _model);

  if (datapointOutput.size() != _expectedOutput.size()) {
    throw std::runtime_error(
      "Expected output amount does not match actual output amount");
  }

  for (size_t j = 0; j < datapointOutput.size(); j++) {
    error += _model.errorFunction(datapointOutput[j],
      _expectedOutput[j]);
  }

  return error;
}

double_t Network::CalculateMeanError(
    const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs,
    const Model &_model) {
  double_t meanError = 0;

  std::vector<double_t> errors;
  std::vector<std::future<double_t>> threads;

  if (_inputs.size() != _expectedOutputs.size()) {
    throw std::runtime_error("Input output pairs not the same amount");
  }

  for (size_t i = 0; i < _inputs.size(); i++) {
    threads.push_back(std::async(std::launch::async, &Network::_CalculateDatapointError, this, _model, _inputs[i], _expectedOutputs[i]));
  }

  for (size_t i = 0; i < threads.size(); i++)
  {
    threads[i].wait();
    errors.push_back(threads[i].get());
  }

  meanError = GLearn::NeuralNetwork::Mean(errors);

  return meanError;
}

std::vector<double_t>
Network::CalculateOutput(const std::vector<double_t> &_inputs,
                         const Model &_model) {
  auto structure = CalculateStructure(_inputs, _model);

  return structure[structure.size() - 1];
}

std::vector<std::vector<double_t>>
Network::CalculateStructure(const std::vector<double_t> &_inputs,
                            const Model &_model) {
  std::vector<std::vector<double_t>> output;
  std::vector<double_t> layerInputs = _inputs;
  std::vector<double_t> layerOutputs;
  double_t neuronOutput;

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
      neuronOutput = _CalculateNeuron(
        layerInputs, _model.weights[i][j], _model.biases[i][j]);

      layerOutputs.push_back(_model.activationFunctions[i][j](neuronOutput));
    }

    layerInputs = layerOutputs;
    output.push_back(layerInputs);
    layerOutputs.clear();
  }

  return output;
}

[[nodiscard]]
std::vector<std::vector<double_t>>
Network::CalculateUnactivatedStructure(const std::vector<double_t>& _inputs, const Model& _model) {
  std::vector<std::vector<double_t>> output;
  std::vector<double_t> layerInputs = _inputs;
  std::vector<double_t> layerOutputs;
  std::vector<double_t> unactivatedOutputs;
  double_t neuronOutput;

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
      neuronOutput = _CalculateNeuron(
        layerInputs, _model.weights[i][j], _model.biases[i][j]);

      layerOutputs.push_back(_model.activationFunctions[i][j](neuronOutput));
      unactivatedOutputs.push_back(neuronOutput);
    }

    layerInputs = layerOutputs;
    output.push_back(unactivatedOutputs);
    layerOutputs.clear();
  }

  return output;
}

Model Network::Learn(const std::vector<std::vector<double_t>> &_inputs,
                     const std::vector<std::vector<double_t>> &_expectedOutputs,
                     Model _model, const size_t _iterations,
                     double_t _learningRate, const double_t _variableLearningReduceRate, const size_t _logInterval) {
  bool loop = false;
  Model temp;
  double_t startMeanError, endMeanError, changeMeanError;

  startMeanError = CalculateMeanError(_inputs, _expectedOutputs, _model);

  if (_iterations == 0)
  {
    loop = true;
  }

  for (size_t i = 0; i < _iterations; i++) {
    temp = _TrainIteration(_inputs, _expectedOutputs, _model, _learningRate);
    endMeanError = CalculateMeanError(_inputs, _expectedOutputs, temp);

    changeMeanError = endMeanError - startMeanError;

    if (changeMeanError > 0 && _variableLearningReduceRate > 0)
    {
      if (_learningRate < 0.0000001)
      {
        break;
      }

      std::cout << '\n';
      std::cout << "Reduced learning rate from " << _learningRate;

      _learningRate *= _variableLearningReduceRate;

      std::cout << " to " << _learningRate << '\n';
      std::cout << '\n';
      continue;
    }

    if (_logInterval > 0)
    {
      if (i % _logInterval == 0)
      {
        std::cout << "Mean Loss: " << endMeanError << '\n';
      }
    }

    startMeanError = endMeanError;
    _model = temp;

    if (loop)
    {
      i = 0;
    }
  }

  return _model;
}
} // namespace NeuralNetwork
} // namespace GLearn
