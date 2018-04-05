/*
 * @author Petr Rek
 * @project CNN library
 * @brief Adam optimizer
 */

#include "src/Optimizers/Adam.h"

/*
 * @brief Sets up parameters
 */
Adam::Adam()
{
	learningRate = 0.001f;
	weightDecay = 0.0f;
	epsilon = 1e-8f;
	b1 = 0.9f;
	b2 = 0.999f;
	b1t = 0.9f;
	b2t = 0.999f;
}

 /*
  * @brief Initializes optimizer with training settings and also creates place to store previous gradients
  */
void Adam::initialize(const unsigned vectorWeightSize, const unsigned vectorsNum, const Dimensions & matrixWeightSize, const unsigned matricesNum)
{
	for (auto i = 0u; i < matricesNum; i++)
	{
		prevMatrixGradientsVector.push_back(Image<BackwardType>(matrixWeightSize));
		prevMatrixGradientsVector.back().clear();

		prevMatrixSquaredGradientsVector.push_back(Image<BackwardType>(matrixWeightSize));
		prevMatrixSquaredGradientsVector.back().clear();
	}

	for (auto i = 0u; i < vectorsNum; i++)
	{
		prevVectorGradientsVector.push_back(std::vector<BackwardType>());
		prevVectorGradientsVector.back().resize(vectorWeightSize);
		std::fill(prevVectorGradientsVector.back().begin(), prevVectorGradientsVector.back().end(), static_cast<BackwardType>(0));

		prevVectorSquaredGradientsVector.push_back(std::vector<BackwardType>());
		prevVectorSquaredGradientsVector.back().resize(vectorWeightSize);
		std::fill(prevVectorSquaredGradientsVector.back().begin(), prevVectorSquaredGradientsVector.back().end(), static_cast<BackwardType>(0));
	}
}


/*
 * @brief Updates weights of single matrix (e.g. FC layer weights)
 */
void Adam::updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize)
{
	static const BackwardType ONE = static_cast<BackwardType>(1.0f);

	const auto flattenedSize = weight.getFlattenedSize();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto batchSizeFactor = static_cast<BackwardType>(1.0f / batchSize);


	for (auto i = 0u; i < flattenedSize; i++)
	{
		auto avgDelta = delta(i) * batchSizeFactor;
 		prevMatrixGradientsVector[0](i) = b1 * prevMatrixGradientsVector[0](i) + (ONE - b1) * avgDelta;
		prevMatrixSquaredGradientsVector[0](i) = b2 * prevMatrixSquaredGradientsVector[0](i) + (ONE - b2) * avgDelta * avgDelta;

		weight(i) -= learningRate / (static_cast<BackwardType>(sqrt(prevMatrixSquaredGradientsVector[0](i) / (ONE - b2t))) + epsilon) * (prevMatrixGradientsVector[0](i) / (ONE - b1t)) + learningWeightWithDecay * weight(i);
	}
	delta.clear();

	b1t *= b1;
	b2t *= b2;
}


/*
 * @brief Updates weights of single vector (e.g. convolutional bias)
 */
void Adam::updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize)
{
	static const BackwardType ONE = static_cast<BackwardType>(1.0f);
	const auto length = delta.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto batchSizeFactor = static_cast<BackwardType>(1.0f / batchSize);

	for (auto i = 0u; i < length; i++)
	{
		auto avgDelta = delta[i] * batchSizeFactor;
		prevVectorGradientsVector[0][i] = b1 * prevVectorGradientsVector[0][i] + (ONE - b1) * avgDelta;
		prevVectorSquaredGradientsVector[0][i] = b2 * prevVectorSquaredGradientsVector[0][i] + (ONE - b2) * avgDelta * avgDelta;

		weight[i] -= learningRate / (static_cast<BackwardType>(sqrt(prevVectorSquaredGradientsVector[0][i] / (ONE - b2t))) + epsilon) * (prevVectorGradientsVector[0][i] / (ONE - b1t)) + learningWeightWithDecay * weight[i];
		delta[i] = 0;
	}

	b1t *= b1;
	b2t *= b2;
}


/*
 * @brief Updates weights of matrices (e.g. convolutional filters)
 */
void Adam::updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize)
{
	static const BackwardType ONE = static_cast<BackwardType>(1.0f);
	const auto flattenedSize = weight[0].getFlattenedSize();
	const auto vectorSize = weight.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto batchSizeFactor = static_cast<BackwardType>(1.0f / batchSize);

	for (auto k = 0u; k < vectorSize; k++)
	{
		for (auto i = 0u; i < flattenedSize; i++)
		{
			auto avgDelta = delta[k](i) * batchSizeFactor;
			prevMatrixGradientsVector[k](i) = b1 * prevMatrixGradientsVector[k](i) + (ONE - b1) * avgDelta;
			prevMatrixSquaredGradientsVector[k](i) = b2 * prevMatrixSquaredGradientsVector[k](i) + (ONE - b2) * avgDelta * avgDelta;

			weight[k](i) -= learningRate / (static_cast<BackwardType>(sqrt(prevMatrixSquaredGradientsVector[k](i) / (ONE - b2t))) + epsilon) * (prevMatrixGradientsVector[k](i) / (ONE - b1t)) + learningWeightWithDecay * weight[k](i);
		}
		delta[k].clear();
	}

	//b1t *= b1; // only update after biases (same iterations)
	//b2t *= b2;
}


/*
 * @brief Clones optimizer
 */
std::unique_ptr<IOptimizer> Adam::clone() const
{
	return std::make_unique<Adam>(*this);
}