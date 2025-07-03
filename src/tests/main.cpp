#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include "GLearn/NeuralNetwork/Model.hpp"
#include "GLearn/NeuralNetwork/Network.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
  GLearn::NeuralNetwork::Network net;
  GLearn::NeuralNetwork::Model model;

  model.weights.push_back({ {0, 0}, {0, 0} });
  model.weights.push_back({ {0, 0}, {0, 0} });
  model.weights.push_back({ {0, 0}, {0, 0} });
  model.weights.push_back({ {0, 0} });

  model.biases.push_back({ 0, 0 });
  model.biases.push_back({ 0, 0 });
  model.biases.push_back({ 0, 0 });
  model.biases.push_back({ 0 });

  model.errorFunction = GLearn::NeuralNetwork::Error::SquaredError;

  model.activationFunctions.push_back(
    { GLearn::NeuralNetwork::Activation::ReLu,
     GLearn::NeuralNetwork::Activation::ReLu });
  model.activationFunctions.push_back({ GLearn::NeuralNetwork::Activation::ReLu , GLearn::NeuralNetwork::Activation::ReLu });
  model.activationFunctions.push_back({ GLearn::NeuralNetwork::Activation::Sigmoid , GLearn::NeuralNetwork::Activation::Sigmoid });
  model.activationFunctions.push_back(
    { GLearn::NeuralNetwork::Activation::Sigmoid });

  model.Randomize();
  model.Print();

  std::cin.get();

  model = net.Learn({ {1, 4}, {3, 2}, {2, 9}, {2, 6}, {5, 4}, {8, 3}, {2, 4} }, { {1}, {1}, {0}, {0}, {1}, {0}, {1} }, model, 100000, 0.2, 0.5);

  auto output = net.CalculateOutput({ 2, 7 }, model);

  for (const auto& out : output) {
    std::cout << out;
  }

  std::cout << std::endl;

  std::cin.get();

  return 0;
}
