#pragma once
#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include "nlohmann/json.hpp"

namespace GLearn {
namespace NeuralNetwork {
struct Model {
  ERROR_FUNCTION errorFunction = nullptr;
  std::vector<std::vector<ACTIVATION_FUNCTION>> activationFunctions;
  std::vector<std::vector<std::vector<double_t>>> weights;
  std::vector<std::vector<double_t>> biases;

  inline void Randomize() {
    std::default_random_engine engine((std::double_t)std::time(nullptr));
    std::uniform_real_distribution<> distribution(0, 1);

    for (size_t layer = 0; layer < weights.size(); layer++)
    {
      for (size_t neuron = 0; neuron < weights[layer].size(); neuron++)
      {
        for (size_t weight = 0; weight < weights[layer][neuron].size(); weight++)
        {
          weights[layer][neuron][weight] = distribution(engine);
        }
      }
    }

    for (size_t layer = 0; layer < biases.size(); layer++)
    {
      for (size_t neuron = 0; neuron < biases[layer].size(); neuron++)
      {
        biases[layer][neuron] = distribution(engine);
      }
    }
  }

  inline void Print() {
    std::cout << "---------------------------------------" << std::endl;

    for (size_t layer = 0; layer < weights.size(); layer++)
    {
      for (size_t neuron = 0; neuron < weights[layer].size(); neuron++)
      {
        std::cout << std::endl;
        std::cout << "Neuron: " << layer << " ; " << neuron << '\n';
        std::cout << "Bias: " << biases[layer][neuron] << '\n';
        std::cout << "Weights: [";

        for (size_t weight = 0; weight < weights[layer][neuron].size(); weight++)
        {
          if (weight == weights[layer][neuron].size() - 1)
          {
            std::cout << weights[layer][neuron][weight] << ']';
          }
          else
          {
            std::cout << weights[layer][neuron][weight] << ", ";
          }
        }

        std::cout << '\n';
        std::cout << std::endl;
      }
    }
  }

  [[nodiscard]]
  inline bool IsValid() const {
    if (errorFunction == nullptr)
      return false;

    if (weights.size() != biases.size())
      return false;

    for (size_t i = 0; i < weights.size(); i++) {
      if (weights[i].size() != biases[i].size())
        return false;

      if (i > 0)
      {
        for (size_t j = 0; j < weights[i].size(); j++) {
          if (weights[i][j].size() != weights[i - 1].size()) {
            return false;
          }
        }
      }
    }

    for (size_t i = 0; i < weights.size(); i++) {
      if (weights.size() != activationFunctions.size())
        return false;

      for (size_t j = 0; j < weights[i].size(); j++) {
        if (weights[i].size() != activationFunctions[i].size())
          return false;
      }
    }

    return true;
  }

  inline void LoadModel(const std::string& _filepath, const std::map<std::string, ERROR_FUNCTION>& _errorFunctionNames = Error::Functions, const std::map<std::string, ACTIVATION_FUNCTION>& _activationFunctionNames = Activation::Functions) {
    nlohmann::json modelJson;
    std::string modelString;
    char c;
    
    std::vector<std::vector<double_t>> layerWeights;
    std::vector<double_t> layerBiases;
    std::vector<ACTIVATION_FUNCTION> layerActivationFunctions;

    std::string activationFunctionString;
    ACTIVATION_FUNCTION activationFunctionPointer;

    std::fstream f;
    f.open(_filepath, std::ios::in);
    if (!f.is_open())
    {
      f.close();
      throw std::runtime_error("Cannot find model file");
    }

    while (!f.eof())
    {
      f.get(c);

      if (!f.eof())
      {
        modelString += c;
      }
    }

    f.close();

    errorFunction = nullptr;
    activationFunctions.clear();
    weights.clear();
    biases.clear();

    modelJson = nlohmann::json::parse(modelString);

    for (const auto& func : _errorFunctionNames)
    {
      if (func.first == modelJson["errorFunction"])
      {
        errorFunction = func.second;
      }
    }

    if (errorFunction == nullptr)
    {
      throw std::runtime_error("Cannot find error function string");
    }

    for (size_t layer = 0; layer < modelJson["layers"].size(); layer++)
    {
      layerWeights.clear();
      layerBiases.clear();
      layerActivationFunctions.clear();

      for (size_t neuron = 0; neuron < modelJson["layers"][std::to_string(layer)]["neurons"].size(); neuron++)
      {
        layerWeights.push_back(modelJson["layers"][std::to_string(layer)]["neurons"][std::to_string(neuron)]["weights"].get<std::vector<double_t>>());
        layerBiases.push_back(modelJson["layers"][std::to_string(layer)]["neurons"][std::to_string(neuron)]["bias"].get<double_t>());
        activationFunctionString = modelJson["layers"][std::to_string(layer)]["neurons"][std::to_string(neuron)]["activationFunction"].get<std::string>();

        activationFunctionPointer = nullptr;
        for (const auto& func : _activationFunctionNames)
        {
          if (func.first == activationFunctionString)
          {
            activationFunctionPointer = func.second;
          }
        }

        if (activationFunctionPointer == nullptr)
        {
          throw std::runtime_error("Cannot find activation function string");
        }

        layerActivationFunctions.push_back(activationFunctionPointer);
      }

      weights.push_back(layerWeights);
      biases.push_back(layerBiases);
      activationFunctions.push_back(layerActivationFunctions);
    }
  }

  inline void SaveModel(const std::string& _filepath, const std::map<std::string, ERROR_FUNCTION> &_errorFunctionNames = Error::Functions, const std::map<std::string, ACTIVATION_FUNCTION>& _activationFunctionNames = Activation::Functions, const size_t _jsonIndentSize = 2) {
    if (!IsValid()) throw std::runtime_error("Cannot save invalid model");

    nlohmann::json modelJson;
    std::string modelString;
    std::string errorFunctionString;
    std::string activationFunctionString;

    for (const auto& func : _errorFunctionNames)
    {
      if (func.second == errorFunction)
      {
        errorFunctionString = func.first;
        break;
      }
    }

    if (errorFunctionString == "")
    {
      throw std::runtime_error("Cannot find error function string");
    }

    modelJson["inputs"] = (weights[0][0]).size();
    modelJson["errorFunction"] = errorFunctionString;

    for (size_t layer = 0; layer < weights.size(); layer++)
    {
      for (size_t neuron = 0; neuron < weights[layer].size(); neuron++)
      {
        for (const auto& func : _activationFunctionNames)
        {
          if (func.second == activationFunctions[layer][neuron])
          {
            activationFunctionString = func.first;
            break;
          }
        }

        if (activationFunctionString == "")
        {
          throw std::runtime_error("Cannot find activation function string");
        }

        modelJson["layers"][std::to_string(layer)]["neurons"][std::to_string(neuron)]["weights"] = weights[layer][neuron];
        modelJson["layers"][std::to_string(layer)]["neurons"][std::to_string(neuron)]["bias"] = biases[layer][neuron];
        modelJson["layers"][std::to_string(layer)]["neurons"][std::to_string(neuron)]["activationFunction"] = activationFunctionString;
      }
    }

    modelString = modelJson.dump(_jsonIndentSize);

    std::fstream f;
    f.open(_filepath, std::ios::out);
    f.write(modelString.c_str(), modelString.length());
    f.close();
  }
};
} // namespace NeuralNetwork
} // namespace GLearn
