/*
 * @author Petr Rek
 * @project CNN library
 * @brief Stochastic gradient descent optimizer
 */

#include "src/Optimizers/Sgd.h"

/*
 * @brief Sets up parameters
 */
Sgd::Sgd()
{
	learningRate = 0.01f;
	weightDecay = 0.0f;
}


 /*
  * @brief Initializes optimizer with training settings, does not need temporary values
  */
void Sgd::initialize(const unsigned , const unsigned , const Dimensions & , const unsigned )
{
}


/*
 * @brief Updates weights of single matrix (e.g. FC layer weights)
 */
void Sgd::updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight.getFlattenedSize();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = static_cast<BackwardType>(learningRate / batchSize);

	for (auto i = 0u; i < flattenedSize; i++)
	{
		weight(i) -= static_cast<BackwardType>(learningRateWithBatchSizeFactor * delta(i) + learningWeightWithDecay * weight(i));
	}
	delta.clear();
}


/*
 * @brief Updates weights of single vector (e.g. cnnvolutional bias)
 */
void Sgd::updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize)
{
	const auto length = delta.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = static_cast<BackwardType>(learningRate / batchSize);

	for (auto i = 0u; i < length; i++)
	{
		weight[i] -= static_cast<BackwardType>(learningRateWithBatchSizeFactor * delta[i] + learningWeightWithDecay * weight[i]);
		delta[i] = 0;
	}
}


/*
 * @brief Updates weights of matrices (e.g. convolutional filters)
 */
void Sgd::updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize)
{
	const auto flattenedSize = weight[0].getFlattenedSize();
	const auto vectorSize = weight.size();
	const auto learningWeightWithDecay = learningRate * weightDecay;
	const auto learningRateWithBatchSizeFactor = static_cast<BackwardType>(learningRate / batchSize);

	for (auto k = 0u; k < vectorSize; k++)
	{
		for (auto i = 0u; i < flattenedSize; i++)
		{
			weight[k](i) -= static_cast<BackwardType>(learningRateWithBatchSizeFactor * delta[k](i) + learningWeightWithDecay * weight[k](i));
		}
		delta[k].clear();
	}
}


/*
 * @brief Clones optimizer
 */
std::unique_ptr<IOptimizer> Sgd::clone() const
{
	return std::make_unique<Sgd>(*this);
}