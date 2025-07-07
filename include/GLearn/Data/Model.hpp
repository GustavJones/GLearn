#pragma once
#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include <cmath>
#include <vector>
#include <string>

namespace GLearn {
namespace Data {
struct Model {
  ERROR_FUNCTION errorFunction = nullptr;
  std::vector<std::vector<ACTIVATION_FUNCTION>> activationFunctions;
  std::vector<std::vector<std::vector<double_t>>> weights;
  std::vector<std::vector<double_t>> biases;

  void Randomize();

  void Print() const;

  [[nodiscard]]
  bool IsValid() const;

  void LoadModel(const std::string& _filepath, const std::map<std::string, ERROR_FUNCTION>& _errorFunctionNames = GLearn::NeuralNetwork::Error::Functions, const std::map<std::string, ACTIVATION_FUNCTION>& _activationFunctionNames = GLearn::NeuralNetwork::Activation::Functions);

  void SaveModel(const std::string& _filepath, const std::map<std::string, ERROR_FUNCTION>& _errorFunctionNames = GLearn::NeuralNetwork::Error::Functions, const std::map<std::string, ACTIVATION_FUNCTION>& _activationFunctionNames = GLearn::NeuralNetwork::Activation::Functions, const size_t _jsonIndentSize = 2);
};
} // namespace Data
} // namespace GLearn
