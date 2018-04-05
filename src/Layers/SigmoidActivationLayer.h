/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Sigmoid activation layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef SIGMOID_ACTIVATION_LAYER_H
#define SIGMOID_ACTIVATION_LAYER_H

#include "src/Layers/ILayer.h"
#include "src/Layers/ActivationLayer.h"

#include "src/Image.h"

/*
 * @brief Sigmoid activation layer
 */
template <class _ForwardType, class _WeightType>
class SigmoidActivationLayer : public ActivationLayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Initializes base class
	 *
	 * @param  input    Dimensions of input matrix
	 */
	SigmoidActivationLayer(const Dimensions & input)
		: ActivationLayer<_ForwardType, _WeightType>(input, ActivationFunction::Sigmoid)
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

		for (auto i = 0u; i < flattenedSize; i++)
		{
			out(i) = static_cast<_ForwardType>(1.0f) / (static_cast<_ForwardType>(1.0f) + static_cast<_ForwardType>(exp(-in(i))));
		}
	}


	/*
	 * @brief Applies activation function derivative and computes output gradients
	 */
	virtual void backwardPropagation(const Image<_ForwardType> &, const Image<_ForwardType> & out, 
		const Image<BackwardType> & inGradients, Image<BackwardType> & outGradients, const TrainingSettings &) override
	{
		auto flattenedSize = out.getFlattenedSize();

		for (auto i = 0u; i < flattenedSize; i++)
		{
			outGradients(i) = (static_cast<BackwardType>(out(i)) * (static_cast<BackwardType>(1.0) - static_cast<BackwardType>(out(i)))) * inGradients(i);
		}
	}

};

#endif