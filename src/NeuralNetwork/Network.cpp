#include "GLearn/NeuralNetwork/Network.hpp"
#include "GLearn/NeuralNetwork/Mean.hpp"
#include "GLearn/Data/Data.hpp"
#include "GLearn/Data/Model.hpp"
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

Data::Model Network::_TrainIteration(
    const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs, Data::Model _model,
    const double_t _learningRate) {
  if (_inputs.size() != _expectedOutputs.size()) {
    throw std::runtime_error("Input output pairs not the same amount");
  }

  _model = _ModifyParameters(_model, _inputs, _expectedOutputs, _learningRate);

  return _model;
}

Data::Model Network::_ModifyParameters(
  Data::Model _model, const std::vector<std::vector<double_t>> &_inputs,
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

Data::Model Network::_ModifyWeight(
  Data::Model _model, const size_t _layer, const size_t _neuron,
    const size_t _weight, const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs, const double_t _learningRate) {
  Data::Model newModel;

  double_t slope; // dError / dWeight
  std::vector<double_t> slopes;
  std::vector<std::future<double_t>> threads;

  // Old prediction based slope
  //double_t dError, dWeight;

  //double_t finalMeanError, startMeanError;

  //dWeight = SLOPE_SIZE;

  //newModel = _model;
  //newModel.weights[_layer][_neuron][_weight] += dWeight;

  //finalMeanError = CalculateMeanError(_inputs, _expectedOutputs, newModel);
  //startMeanError = CalculateMeanError(_inputs, _expectedOutputs, _model);

  //dError = finalMeanError - startMeanError;;

  //slope = dError / dWeight;

  for (size_t i = 0; i < _inputs.size(); i++)
  {
    threads.push_back(std::async(std::launch::async, &Network::_CalculateWeightCostDerivative, this, _layer, _neuron, _weight, _inputs[i], _expectedOutputs[i], _model));
    //slopes.push_back(_CalculateWeightCostDerivative(_layer, _neuron, _weight, _inputs[i], _expectedOutputs[i], _model));
  }

  for (size_t i = 0; i < threads.size(); i++)
  {
    threads[i].wait();
    slopes.push_back(threads[i].get());
  }

  slope = GLearn::NeuralNetwork::Mean(slopes);

  _model.weights[_layer][_neuron][_weight] -= slope * _learningRate;
  return _model;
}

Data::Model Network::_ModifyBias(
  Data::Model _model, const size_t _layer, const size_t _neuron,
    const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs, 
    const double_t _learningRate) {
  Data::Model newModel;

  double_t slope; // dError / dWeight
  std::vector<double_t> slopes;
  std::vector<std::future<double_t>> threads;

  // Old prediction based slope
  //double_t dError, dBias;

  //double_t finalMeanError, startMeanError;

  //dBias = SLOPE_SIZE;

  //newModel = _model;
  //newModel.biases[_layer][_neuron] += dBias;

  //auto finalThread = std::async(std::launch::async, &Network::CalculateMeanError, this, _inputs, _expectedOutputs, newModel);
  //auto startThread = std::async(std::launch::async, &Network::CalculateMeanError, this, _inputs, _expectedOutputs, _model);

  //finalThread.wait();
  //startThread.wait();

  //finalMeanError = finalThread.get();
  //startMeanError = startThread.get();

  //dError = finalMeanError - startMeanError;

  //slope = dError / dBias;

  for (size_t i = 0; i < _inputs.size(); i++)
  {
    threads.push_back(std::async(std::launch::async, &Network::_CalculateBiasCostDerivative, this, _layer, _neuron, _inputs[i], _expectedOutputs[i], _model));
    //slopes.push_back(_CalculateBiasCostDerivative(_layer, _neuron, _inputs[i], _expectedOutputs[i], _model));
  }

  for (size_t i = 0; i < threads.size(); i++)
  {
    threads[i].wait();
    slopes.push_back(threads[i].get());
  }

  slope = GLearn::NeuralNetwork::Mean(slopes);

  _model.biases[_layer][_neuron] -= slope * _learningRate;
  return _model;
}

double_t Network::_CalculateDatapointError(const Data::Model &_model, const std::vector<double_t> &_input, const std::vector<double_t> &_expectedOutput) {
  double_t error = 0;

  std::vector<double_t> datapointOutput;
  datapointOutput = CalculateOutput(_input, _model);

  if (datapointOutput.size() != _expectedOutput.size()) {
    throw std::runtime_error(
      "Expected output amount does not match actual output amount");
  }

  for (size_t j = 0; j < datapointOutput.size(); j++) {
    error += _model.errorFunction(datapointOutput[j],
      _expectedOutput[j], false);
  }

  return error;
}

Derivatives Network::_CalculateDeltas(const Data::Model& _model, const std::vector<double_t>& _input, const std::vector<double_t>& _expectedOutput) {
  Derivatives out;
  std::vector<std::vector<double_t>> unactivatedStructure = CalculateUnactivatedStructure(_input, _model);
  std::vector<std::vector<double_t>> activatedStructure = CalculateStructure(_input, _model);

  out.deltas.resize(_model.weights.size());
  out.biasDeltas.resize(_model.weights.size());
  out.weightDeltas.resize(_model.weights.size());

  double_t sum;
  size_t lastLayer = _model.weights.size() - 1;

  out.deltas[lastLayer].resize(_model.weights[lastLayer].size());
  out.biasDeltas[lastLayer].resize(_model.weights[lastLayer].size());
  out.weightDeltas[lastLayer].resize(_model.weights[lastLayer].size());

  for (size_t neuron = 0; neuron < _model.weights[lastLayer].size(); neuron++)
  {
    out.weightDeltas[lastLayer][neuron].resize(_model.weights[lastLayer][neuron].size());

    out.deltas[lastLayer][neuron] = _model.errorFunction(activatedStructure[lastLayer][neuron], _expectedOutput[neuron], true) * _model.activationFunctions[lastLayer][neuron](unactivatedStructure[lastLayer][neuron], true);
    out.biasDeltas[lastLayer][neuron] = out.deltas[lastLayer][neuron];

    for (size_t weight = 0; weight < out.weightDeltas[lastLayer][neuron].size(); weight++)
    {
      if (lastLayer > 0)
      {
        out.weightDeltas[lastLayer][neuron][weight] = out.deltas[lastLayer][neuron] * activatedStructure[lastLayer - 1][weight];
      }
      else
      {
        out.weightDeltas[lastLayer][neuron][weight] = out.deltas[lastLayer][neuron] * _input[weight];
      }      
    }
  }

  for (int64_t layer = lastLayer - 1; layer >= 0; layer--)
  {
    out.deltas[layer].resize(_model.weights[layer].size());
    out.biasDeltas[layer].resize(_model.weights[layer].size());
    out.weightDeltas[layer].resize(_model.weights[layer].size());

    for (size_t neuron = 0; neuron < _model.weights[layer].size(); neuron++)
    {
      out.weightDeltas[layer][neuron].resize(_model.weights[layer][neuron].size());

      sum = 0.0;
      for (size_t i = 0; i < _model.weights[layer + 1].size(); i++)
      {
        sum += _model.weights[layer + 1][i][neuron] * out.deltas[layer + 1][i];
      }

      out.deltas[layer][neuron] = _model.activationFunctions[layer][neuron](unactivatedStructure[layer][neuron], true) * sum;
      out.biasDeltas[layer][neuron] = out.deltas[layer][neuron];

      for (size_t weight = 0; weight < out.weightDeltas[layer][neuron].size(); weight++)
      {
        if (layer > 0)
        {
          out.weightDeltas[layer][neuron][weight] = out.deltas[layer][neuron] * activatedStructure[layer - 1][weight];
        }
        else
        {
          out.weightDeltas[layer][neuron][weight] = out.deltas[layer][neuron] * _input[weight];
        }        
      }
    }
  }

  return out;
}

double_t Network::_CalculateBiasCostDerivative(const size_t _layer, const size_t _neuron, const std::vector<double_t>& _input,
  const std::vector<double_t>& _expectedOutput,
  const Data::Model& _model) {
  const std::vector<std::vector<double_t>> unactivatedStructure = CalculateUnactivatedStructure(_input, _model);
  const std::vector<std::vector<double_t>> activatedStructure = CalculateStructure(_input, _model);

  double_t biasDerivative, activationDerivative, parentDerivative;
  biasDerivative = 1.0;
  activationDerivative = _model.activationFunctions[_layer][_neuron](unactivatedStructure[_layer][_neuron], true);

  parentDerivative = _CalculateParentDerivative(_layer, _neuron, _model, activatedStructure, unactivatedStructure, _expectedOutput);

  return biasDerivative * activationDerivative * parentDerivative;
}

double_t Network::_CalculateWeightCostDerivative(const size_t _layer, const size_t _neuron, const size_t _weight, const std::vector<double_t>& _input,
  const std::vector<double_t>& _expectedOutput,
  const Data::Model& _model) {
  const std::vector<std::vector<double_t>> unactivatedStructure = CalculateUnactivatedStructure(_input, _model);
  const std::vector<std::vector<double_t>> activatedStructure = CalculateStructure(_input, _model);

  double_t weightDerivative, activationDerivative, parentDerivative;

  if (_layer > 0)
  {
    weightDerivative = activatedStructure[_layer - 1][_weight];
  }
  else
  {
    weightDerivative = _input[_weight];
  }
  
  activationDerivative = _model.activationFunctions[_layer][_neuron](unactivatedStructure[_layer][_neuron], true);

  parentDerivative = _CalculateParentDerivative(_layer, _neuron, _model, activatedStructure, unactivatedStructure, _expectedOutput);

  return weightDerivative * activationDerivative * parentDerivative;
}

double_t Network::_CalculateParentDerivative(const size_t _layer, const size_t _neuron, const Data::Model& _model, const std::vector<std::vector<double_t>>& _activatedStructure, const std::vector<std::vector<double_t>>& _unactivatedStructure, const std::vector<double_t>& _expectedOutput) {
  const size_t nextLayer = _layer + 1;
  double_t derivative = 0;
  double_t branchDerivative;

  if (nextLayer >= _model.weights.size())
  {
    return _model.errorFunction(_activatedStructure[_layer][_neuron], _expectedOutput[_neuron], true);
  }

  // For every connected neuron
  for (size_t i = 0; i < _model.weights[nextLayer].size(); i++)
  {
    // dy(current) / da(previous)
    branchDerivative = _model.weights[nextLayer][i][_neuron];

    // da(current) / dy(current)
    branchDerivative *= _model.activationFunctions[nextLayer][i](_unactivatedStructure[nextLayer][i], true);

    // dC / da(current)
    branchDerivative *= _CalculateParentDerivative(nextLayer, i, _model, _activatedStructure, _unactivatedStructure, _expectedOutput);

    // Sum of different neuron weights because z = w*a1 + w*a2 + ... + b so sum of derivatives
    derivative += branchDerivative;
  }

  return derivative;
}

double_t Network::CalculateMeanError(
    const std::vector<std::vector<double_t>> &_inputs,
    const std::vector<std::vector<double_t>> &_expectedOutputs,
    const Data::Model &_model) {
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
                         const Data::Model &_model) {
  auto structure = CalculateStructure(_inputs, _model);

  return structure[structure.size() - 1];
}

std::vector<std::vector<double_t>>
Network::CalculateStructure(const std::vector<double_t> &_inputs,
                            const Data::Model &_model) {
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

      layerOutputs.push_back(_model.activationFunctions[i][j](neuronOutput, false));
    }

    layerInputs = layerOutputs;
    output.push_back(layerInputs);
    layerOutputs.clear();
  }

  return output;
}

[[nodiscard]]
std::vector<std::vector<double_t>>
Network::CalculateUnactivatedStructure(const std::vector<double_t>& _inputs, const Data::Model& _model) {
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

      layerOutputs.push_back(_model.activationFunctions[i][j](neuronOutput, false));
      unactivatedOutputs.push_back(neuronOutput);
    }

    layerInputs = layerOutputs;
    output.push_back(unactivatedOutputs);
    unactivatedOutputs.clear();
    layerOutputs.clear();
  }

  return output;
}

Data::Model Network::Learn(const std::vector<std::vector<double_t>> &_inputs,
                     const std::vector<std::vector<double_t>> &_expectedOutputs,
                     Data::Model _model, const size_t _epochIterations, size_t _batchCount,
                     double_t _learningRate, const double_t _variableLearningReduceRate, const size_t _logInterval) {
  bool loop = false;
  Data::Model temp;
  double_t startMeanError, endMeanError, changeMeanError;
  std::vector<std::vector<double_t>> batchInputs, batchExpectedOutputs;

  if (_batchCount < 1)
  {
    _batchCount = 1;
  }

  startMeanError = CalculateMeanError(_inputs, _expectedOutputs, _model);

  if (_epochIterations == 0)
  {
    loop = true;
  }

  for (size_t i = 0; i < _epochIterations; i++) {
    temp = _model;

    for (size_t j = 0; j < _batchCount; j++)
    {
      batchInputs = GLearn::Data::SplitDataset(_inputs, _batchCount, j);
      batchExpectedOutputs = GLearn::Data::SplitDataset(_expectedOutputs, _batchCount, j);
      temp = _TrainIteration(batchInputs, batchExpectedOutputs, temp, _learningRate);

      if (_logInterval > 0 && j > 0)
      {
        if (j % _logInterval == 0)
        {
          std::cout << "Completed batch: " << j << std::endl;
        }
      }
    }

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

    if (i % 10 == 0)
    {
      std::cout << "Mean Loss: " << endMeanError << '\n';
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
