/*
 * @author Petr Rek
 * @project CNN library
 * @brief Stochastic gradient descent optimizer with momentum
 */

#include "src/Optimizers/SgdWithMomentum.h"

/*
 * @brief Sets up parameters
 */
SgdWithMomentum::SgdWithMomentum()
{
	learningRate = 0.01f;
	momentum = 0.9f;
	weightDecay = 0.0f;
}


 /*
  * @brief Initializes optimizer with training settings and also creates place to store previous gradients
  */
void SgdWithMomentum::initialize(const unsigned vectorWeightSize, 
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
void SgdWithMomentum::updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight.getFlattenedSize();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = static_cast<BackwardType>(learningRate / batchSize);

	for (auto i = 0u; i < flattenedSize; i++)
	{
		prevMatrixGradientsVector[0](i) = momentum * prevMatrixGradientsVector[0](i) + learningRateWithBatchSizeFactor * delta(i);
		weight(i) -= prevMatrixGradientsVector[0](i) + learningWeightWithDecay * weight(i);
	}
	delta.clear();
}


/*
 * @brief Updates weights of single vector (e.g. convolutional bias)
 */
void SgdWithMomentum::updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize)
{
	const auto length = delta.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = static_cast<BackwardType>(learningRate / batchSize);

	for (auto i = 0u; i < length; i++)
	{
		prevVectorGradientsVector[0][i] = momentum * prevVectorGradientsVector[0][i] + learningRateWithBatchSizeFactor * delta[i];
		weight[i] -= prevVectorGradientsVector[0][i] + learningWeightWithDecay * weight[i];
		delta[i] = 0;
	}
}


/*
 * @brief Updates weights of matrices (e.g. convolutional filters)
 */
void SgdWithMomentum::updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight[0].getFlattenedSize();
	const auto vectorSize = weight.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = static_cast<BackwardType>(learningRate / batchSize);

	for (auto k = 0u; k < vectorSize; k++)
	{
		for (auto i = 0u; i < flattenedSize; i++)
		{
			prevMatrixGradientsVector[k](i) = momentum * prevMatrixGradientsVector[k](i) + learningRateWithBatchSizeFactor * delta[k](i);
			weight[k](i) -= prevMatrixGradientsVector[k](i) + learningWeightWithDecay * weight[k](i);
		}
		delta[k].clear();
	}
}


/*
 * @brief Clones optimizer
 */
std::unique_ptr<IOptimizer> SgdWithMomentum::clone() const
{
	return std::make_unique<SgdWithMomentum>(*this);
}