#pragma once
#include <vector>

namespace GLearn {
namespace Data {
  [[nodiscard]]
  std::vector<std::vector<double_t>> SplitDataset(const std::vector<std::vector<double_t>> _data, const size_t _segmentCount, const size_t _segmentIndex);
}
}