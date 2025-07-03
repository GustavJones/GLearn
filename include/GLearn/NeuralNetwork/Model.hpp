#pragma once
#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

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
};
} // namespace NeuralNetwork
} // namespace GLearn
