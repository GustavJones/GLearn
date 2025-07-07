#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include "GLearn/Data/Model.hpp"
#include "GLearn/NeuralNetwork/Network.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
  GLearn::NeuralNetwork::Network net;
  GLearn::Data::Model model;

  try
  {
    model.LoadModel("checkpoint.model");
  }
  catch (const std::exception& e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;

    model.weights = { { {1, 1}, {1, 1} }, { {1, 1}, {1, 1} }, { {1, 1}, {1, 1} }, { {1, 1}, {1, 1} } };
    model.biases = { { 0, 0 } , { 0, 0 } , { 0, 0 }, { 0, 0 } };
    model.activationFunctions = { { GLearn::NeuralNetwork::Activation::Sigmoid,
     GLearn::NeuralNetwork::Activation::Sigmoid }, { GLearn::NeuralNetwork::Activation::ReLu , GLearn::NeuralNetwork::Activation::ReLu }, { GLearn::NeuralNetwork::Activation::Sigmoid , GLearn::NeuralNetwork::Activation::Sigmoid }, { GLearn::NeuralNetwork::Activation::Sigmoid, GLearn::NeuralNetwork::Activation::Sigmoid } };
    model.errorFunction = GLearn::NeuralNetwork::Error::AbsoluteError;

    model.Randomize();
  }

  if (!model.IsValid())
  {
    throw std::runtime_error("Invalid model");
  }

  model = net.Learn({ {1, 4}, {3, 2}, {2, 9}, {2, 6}, {5, 4}, {8, 3}, {2, 4}, {5, 8} }, { {1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 1}, {0, 1}, {1, 0}, {1, 1} }, model, 1000, 8, 0.1, 0.9);

  model.SaveModel("checkpoint.model");

  auto output = net.CalculateOutput({ 5, 4 }, model);

  for (const auto& out : output) {
    std::cout << out << ' ';
  }

  std::cout << std::endl;

  std::cin.get();

  return 0;
}
