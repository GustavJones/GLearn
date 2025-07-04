#include "GLearn/NeuralNetwork/ActivationFunctions.hpp"
#include "GLearn/NeuralNetwork/ErrorFunctions.hpp"
#include "GLearn/NeuralNetwork/Model.hpp"
#include "GLearn/NeuralNetwork/Network.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
  GLearn::NeuralNetwork::Network net;
  GLearn::NeuralNetwork::Model model;

  try
  {
    model.LoadModel("checkpoint.model");
  }
  catch (const std::exception& e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;

    model.weights = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} }, { {0, 0} } };
    model.biases = { { 0, 0 } , { 0, 0 } , { 0, 0 }, { 0 } };
    model.activationFunctions = { { GLearn::NeuralNetwork::Activation::ReLu,
     GLearn::NeuralNetwork::Activation::ReLu }, { GLearn::NeuralNetwork::Activation::ReLu , GLearn::NeuralNetwork::Activation::ReLu }, { GLearn::NeuralNetwork::Activation::Sigmoid , GLearn::NeuralNetwork::Activation::Sigmoid }, { GLearn::NeuralNetwork::Activation::Sigmoid } };
    model.errorFunction = GLearn::NeuralNetwork::Error::SquaredError;

    model.Randomize();
  }

  model.Print();

  std::cin.get();

  model = net.Learn({ {1, 4}, {3, 2}, {2, 9}, {2, 6}, {5, 4}, {8, 3}, {2, 4} }, { {1}, {1}, {0}, {0}, {1}, {0}, {1} }, model, 100000, 0.8, 0.5);

  model.SaveModel("checkpoint.model");

  auto output = net.CalculateOutput({ 5, 4 }, model);

  for (const auto& out : output) {
    std::cout << out;
  }

  std::cout << std::endl;

  std::cin.get();

  return 0;
}
