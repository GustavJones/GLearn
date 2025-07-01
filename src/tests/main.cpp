#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include "GLearn/NeuralNetwork/Model.hpp"
#include "GLearn/NeuralNetwork/Network.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
  GLearn::NeuralNetwork::Network net;
  GLearn::NeuralNetwork::Model model;

  model.weights.push_back({{1, 1}, {1, 1}});
  model.weights.push_back({{1, 1}});

  model.biases.push_back({0, 0});
  model.biases.push_back({0});

  model.errorFunction = GLearn::NeuralNetwork::Error::SquaredError;
  model.activationFunctions.push_back(
      {GLearn::NeuralNetwork::Activation::None,
       GLearn::NeuralNetwork::Activation::None});
  model.activationFunctions.push_back(
      {GLearn::NeuralNetwork::Activation::None});

  net.Learn({{1, 1}, {1, 2}, {2, 1}, {2, 2}}, {{2}, {3}, {2}, {1}}, model);

  auto output = net.Calculate({1, 1}, model);

  for (const auto &out : output) {
    std::cout << out;
  }

  std::cout << std::endl;

  return 0;
}
