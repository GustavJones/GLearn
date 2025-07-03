#include "GLearn/NeuralNetwork/Mean.hpp"
#include <stdexcept>

namespace GLearn {
namespace NeuralNetwork {
double_t GLearn::NeuralNetwork::Mean(const std::vector<double_t> &_values, bool _derivative) {
  double_t sum = 0;

  if (_derivative)
  {
    return 1.0 / _values.size();
  }

  if (_values.size() <= 0)
  {
    throw std::runtime_error("No values provided for mean");
  }

  for (double_t val : _values)
  {
    sum += val;
  }

  sum /= _values.size();
  return sum;
}
} // namespace NeuralNetwork
} // namespace GLearn
