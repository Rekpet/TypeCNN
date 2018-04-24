/*
 * @author Petr Rek
 * @project CNN library
 * @brief Stochastic gradient descent optimizer with nestorov momentum
 */

#include "src/Optimizers/SgdWithNestorovMomentum.h"


/*
 * @brief Sets up parameters
 */
SgdWithNestorovMomentum::SgdWithNestorovMomentum()
{
	learningRate = 0.01f;
	momentum = 0.9f;
	weightDecay = 0.0f;
}


 /*
  * @brief Initializes optimizer with training settings and also creates place to store previous gradients
  */
void SgdWithNestorovMomentum::initialize(const unsigned vectorWeightSize, 
	const unsigned vectorsNum, const Dimensions & matrixWeightSize, const unsigned matricesNum)
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
void SgdWithNestorovMomentum::updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight.getFlattenedSize();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = learningRate / static_cast<BackwardType>(static_cast<float>(batchSize));

	for (auto i = 0u; i < flattenedSize; i++)
	{
		auto tmp = momentum * prevMatrixGradientsVector[0](i) + learningRateWithBatchSizeFactor * delta(i);
		weight(i) -= (- momentum) * prevMatrixGradientsVector[0](i) + (static_cast<BackwardType>(1) + momentum) * tmp + learningWeightWithDecay * weight(i);
		prevMatrixGradientsVector[0](i) = tmp;
	}
	delta.clear();
}


/*
 * @brief Updates weights of single vector (e.g. convolutional bias)
 */
void SgdWithNestorovMomentum::updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize)
{
	const auto length = delta.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = learningRate / static_cast<BackwardType>(static_cast<float>(batchSize));

	for (auto i = 0u; i < length; i++)
	{
		auto tmp = momentum * prevVectorGradientsVector[0][i] + learningRateWithBatchSizeFactor * delta[i];
		weight[i] -= (- momentum) * prevVectorGradientsVector[0][i] + (static_cast<BackwardType>(1) + momentum) * tmp + learningWeightWithDecay * weight[i];
		prevVectorGradientsVector[0][i] = tmp;
		delta[i] = 0;
	}
}


/*
 * @brief Updates weights of matrices (e.g. convolutional filters)
 */
void SgdWithNestorovMomentum::updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight[0].getFlattenedSize();
	const auto vectorSize = weight.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = learningRate / static_cast<BackwardType>(static_cast<float>(batchSize));

	for (auto k = 0u; k < vectorSize; k++)
	{
		for (auto i = 0u; i < flattenedSize; i++)
		{
			auto tmp = momentum * prevMatrixGradientsVector[k](i) + learningRateWithBatchSizeFactor * delta[k](i);
			weight[k](i) -= (- momentum) * prevMatrixGradientsVector[k](i) + (static_cast<BackwardType>(1) + momentum) * tmp + learningWeightWithDecay * weight[k](i);
			prevMatrixGradientsVector[k](i) = tmp;
		}
		delta[k].clear();
	}
}


/*
 * @brief Clones optimizer
 */
std::unique_ptr<IOptimizer> SgdWithNestorovMomentum::clone() const
{
	return std::make_unique<SgdWithNestorovMomentum>(*this);
}