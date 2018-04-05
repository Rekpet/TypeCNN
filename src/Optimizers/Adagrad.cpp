/*
 * @author Petr Rek
 * @project CNN library
 * @brief Adagrad optimizer
 */

#include "src/Optimizers/Adagrad.h"

/*
 * @brief Sets up parameters
 */
Adagrad::Adagrad()
{
	learningRate = 0.01f;
	weightDecay = 0.0f;
	epsilon = 1e-8f;
}

 /*
  * @brief Initializes optimizer with training settings and also creates place to store previous gradients
  */
void Adagrad::initialize(const unsigned vectorWeightSize, const unsigned vectorsNum, const Dimensions & matrixWeightSize, const unsigned matricesNum)
{
	for (auto i = 0u; i < matricesNum; i++)
	{
		prevMatrixGradientsVector.push_back(Image<BackwardType>(matrixWeightSize));
		prevMatrixGradientsVector.back().clear();
	}

	for (auto i = 0u; i < vectorsNum; i++)
	{
		prevVectorGradientsVector.push_back(std::vector<BackwardType>());
		prevVectorGradientsVector.back().resize(vectorWeightSize);
		std::fill(prevVectorGradientsVector.back().begin(), prevVectorGradientsVector.back().end(), static_cast<BackwardType>(0));
	}
}


/*
 * @brief Updates weights of single matrix (e.g. FC layer weights)
 */
void Adagrad::updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight.getFlattenedSize();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto batchSizeFactor = static_cast<BackwardType>(1.0f / batchSize);

	for (auto i = 0u; i < flattenedSize; i++)
	{
		auto avgDelta = delta(i) * batchSizeFactor;
		prevMatrixGradientsVector[0](i) += avgDelta * avgDelta;
		weight(i) -= learningRate * avgDelta / static_cast<BackwardType>(sqrt(prevMatrixGradientsVector[0](i) + epsilon)) + learningWeightWithDecay * weight(i);
	}
	delta.clear();
}


/*
 * @brief Updates weights of single vector (e.g. convolutional bias)
 */
void Adagrad::updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize)
{
	const auto length = delta.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto batchSizeFactor = static_cast<BackwardType>(1.0f / batchSize);

	for (auto i = 0u; i < length; i++)
	{
		auto avgDelta = delta[i] * batchSizeFactor;
		prevVectorGradientsVector[0][i] += avgDelta * avgDelta;
		weight[i] -= learningRate * avgDelta / static_cast<BackwardType>(sqrt(prevVectorGradientsVector[0][i] + epsilon)) + learningWeightWithDecay * weight[i];
		delta[i] = 0;
	}
}


/*
 * @brief Updates weights of matrices (e.g. convolutional filters)
 */
void Adagrad::updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight[0].getFlattenedSize();
	const auto vectorSize = weight.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto batchSizeFactor = static_cast<BackwardType>(1.0f / batchSize);

	for (auto k = 0u; k < vectorSize; k++)
	{
		for (auto i = 0u; i < flattenedSize; i++)
		{
			auto avgDelta = delta[k](i) * batchSizeFactor;
			prevMatrixGradientsVector[k](i) += avgDelta * avgDelta;
			weight[k](i) -= learningRate * avgDelta / static_cast<BackwardType>(sqrt(prevMatrixGradientsVector[k](i) + epsilon)) + learningWeightWithDecay * weight[k](i);
		}
		delta[k].clear();
	}
}

/*
 * @brief Clones optimizer
 */
std::unique_ptr<IOptimizer> Adagrad::clone() const
{
	return std::make_unique<Adagrad>(*this);
}