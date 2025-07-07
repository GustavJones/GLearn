#pragma once
#include <vector>

namespace GLearn {
namespace NeuralNetwork {
	struct Derivatives
	{
		std::vector<std::vector<double_t>> deltas;
		std::vector<std::vector<std::vector<double_t>>> weightDeltas;
		std::vector<std::vector<double_t>> biasDeltas;
	};
}
}