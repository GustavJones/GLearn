#include "GLearn/Data/Data.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace GLearn {
namespace Data {
  std::vector<std::vector<double_t>> SplitDataset(const std::vector<std::vector<double_t>> _data, const size_t _segmentCount, const size_t _segmentIndex) {
    std::vector<std::vector<double_t>> output;

    if (_data.size() % _segmentCount != 0)
    {
      throw std::runtime_error("Data cannot be split in " + std::to_string(_segmentCount) + " segments without remainder");
    }

    const size_t segmentSize = _data.size() / _segmentCount;
    
    output.reserve(segmentSize);
    for (size_t i = 0; i < segmentSize; i++)
    {
      output.push_back(_data[(_segmentIndex * segmentSize) + i]);
    }

    return output;
  }
}
}