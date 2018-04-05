/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Universal dropout layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "src/Layers/ILayer.h"

#include "src/Image.h"

 /*
  * @brief Exception thrown if problems occur during layer initialization
  */
class DropoutLayerException : public CNNException
{
	using CNNException::CNNException;
};

/*
 * @brief Dropout layer
 */
template <class _ForwardType, class _WeightType>
class DropoutLayer : public ILayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Initializes drop out layer and checks parameters
	 *
	 * @param  input         Dimensions if input matrix
	 * @param  probability   Probability with which each pixel may be reset to zero
	 */
	DropoutLayer(const Dimensions & input, const float probability = 0.01f)
		: inputSize(input)
		, outputSize(input)
		, probability(probability)
		, dropoutHistory(input)
		, output(input)
		, gradientOutput(input)
	{
		if (probability < 0 || probability > 1)
		{
			throw DropoutLayerException("Dropout probability must be in range <0,1>.");
		}

		// Dropout layer prevents overfitting, it has no use during predictions
		this->useOnlyWhenLearning = true;
	}


	/*
	 * @brief Propagates matrix, zeroes out some pixels along the way
	 */
	virtual void forwardPropagation(const Image<_ForwardType> & in, Image<_ForwardType> & out) override
	{
		if (in.getDimensions() != inputSize)
		{
			throw InputImageDoesNotHaveCorrectDimensions("Input image had different dimensions than declared when initializing Dropout layer.");
		}

		out = in;

		dropoutHistory.clear();

		// Clear some pixels and save their position (for back propagation)
		auto flattenedSize = out.getFlattenedSize();
		for (auto i = 0u; i < flattenedSize; i++)
		{
			auto randNumber = static_cast<float>(rand()) / RAND_MAX;
			if (randNumber < probability)
			{
				out(i) = static_cast<_ForwardType>(0);
				dropoutHistory(i) = 1;
			}
		}
	}


	/*
	 * @brief Initializes drop out layer and checks parameters
	 */
	virtual void backwardPropagation(const Image<_ForwardType> &, const Image<_ForwardType> &, 
		const Image<BackwardType> & inGradients, Image<BackwardType> & outGradients, const TrainingSettings &) override
	{
		outGradients = inGradients;

		auto flattenedSize = outGradients.getFlattenedSize();

		// Only propagate gradients for pixels that were untouched
		for (auto i = 0u; i < flattenedSize; i++)
		{
			if (dropoutHistory(i) == 1)
			{
				outGradients(i) = 0.0f;
			}
		}
	}


	/*
	 * @brief Returns expected input size
	 */
	virtual Dimensions getInputSize() const override
	{
		return inputSize;
	}


	/*
	 * @brief Returns output size
	 */
	virtual Dimensions getOutputSize() const override
	{
		return outputSize;
	}


	/*
	 * @brief Returns a reference to layer output
	 */
	virtual Image<_ForwardType> & getOutput() override
	{
		return output;
	}

	/*
	 * @brief Returns a reference to layer gradient output
	 */
	virtual Image<BackwardType> & getGradientOutput() override
	{
		return gradientOutput;
	}

	/*
	 * @brief Returns dropout probability
	 */
	float getDropoutProbability() const
	{
		return probability;
	}

private:

	/// Accepted input size
	Dimensions inputSize;

	/// Output size
	Dimensions outputSize;

	/// Probability of dropout
	float probability;

	/// Contains history of dropped pixels for backpropagation
	Image<unsigned> dropoutHistory;

	/// Output to be forward propagated to next layer
	Image<_ForwardType> output;

	/// Gradients to be backward propagated to previous layer
	Image<BackwardType> gradientOutput;

};

#endif