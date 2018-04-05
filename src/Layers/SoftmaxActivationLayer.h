/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Softmax activation layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef SOFTMAX_ACTIVATION_LAYER_H
#define SOFTMAX_ACTIVATION_LAYER_H

#include "src/Layers/ILayer.h"
#include "src/Layers/ActivationLayer.h"
#include "src/Image.h"
#include "src/Utils/Limits.h"

 /*
  * @brief Softmax activation layer
  */
template <class _ForwardType, class _WeightType>
class SoftmaxActivationLayer : public ActivationLayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Initializes base class
	 *
	 * @param  input    Dimensions of input matrix
	 */
	SoftmaxActivationLayer(const Dimensions & input)
		: ActivationLayer<_ForwardType, _WeightType>(input, ActivationFunction::SoftMax)
	{
	}


	/*
	 * @brief Applies activation functions on all cells of input matrix
	 */
	virtual void forwardPropagation(const Image<_ForwardType> & in, Image<_ForwardType> & out) override
	{
		if (in.getDimensions() != this->inputSize)
		{
			throw InputImageDoesNotHaveCorrectDimensions("Input to Activation layer has different dimensions than declared during initilization.");
		}

		auto flattenedSize = in.getFlattenedSize();

		// Find maximum
		softmaxMax = Limits::getMinimumValue<_ForwardType>();
		for (auto i = 0u; i < flattenedSize; i++)
		{
			if (in(i) > softmaxMax)
			{
				softmaxMax = in(i);
			}
		}

		// Compute sum
		softmaxSum = 0;
		for (auto i = 0u; i < flattenedSize; i++)
		{
			softmaxSum += static_cast<_ForwardType>(exp(in(i) - softmaxMax));
		}

		// Compute output values
		for (auto i = 0u; i < flattenedSize; i++)
		{
			out(i) = static_cast<_ForwardType>(exp(in(i) - softmaxMax)) / softmaxSum;
		}
	}


	/*
	 * @brief Applies activation function derivative and computes output gradients
	 */
	virtual void backwardPropagation(const Image<_ForwardType> &, const Image<_ForwardType> & out, 
		const Image<BackwardType> & inGradients, Image<BackwardType> & outGradients, const TrainingSettings &) override
	{
		auto flattenedSize = out.getFlattenedSize();

		outGradients.clear();
		for (auto i = 0u; i < flattenedSize; i++)
		{
			for (auto k = 0u; k < flattenedSize; k++)
			{
				// https://stackoverflow.com/questions/37790990/derivative-of-a-softmax-function-explanation
				if (i == k)
				{
					outGradients(i) += (static_cast<BackwardType>(out(i)) * (static_cast<BackwardType>(1) - static_cast<BackwardType>(out(i)))) * inGradients(k);
				}
				else
				{
					outGradients(i) += (-static_cast<BackwardType>(out(i)) * static_cast<BackwardType>(out(k))) * inGradients(k);
				}
			}
		}
	}

private:

	/// Sum used by softmax activation
	_ForwardType softmaxSum;

	/// Max used by softmax activation
	_ForwardType softmaxMax;

};

#endif