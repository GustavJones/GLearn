#pragma once
#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include <cmath>
#include <vector>

namespace GLearn {
namespace NeuralNetwork {
struct Model {
  ERROR_FUNCTION errorFunction;
  std::vector<std::vector<ACTIVATION_FUNCTION>> activationFunctions;
  std::vector<std::vector<std::vector<double_t>>> weights;
  std::vector<std::vector<double_t>> biases;

  [[nodiscard]]
  inline bool IsValid() const {
    if (errorFunction == nullptr)
      return false;

    if (weights.size() != biases.size())
      return false;

    for (size_t i = 1; i < weights.size(); i++) {
      if (weights[i].size() != biases[i].size())
        return false;

      for (size_t j = 0; j < weights[i].size(); j++) {
        if (weights[i][j].size() != weights[i - 1].size()) {
          return false;
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
