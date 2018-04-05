/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Universal activation layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "src/Layers/ILayer.h"

#include "src/Image.h"


/*
 * @brief Exception thrown if problems occur
 */
class ActivationLayerException : public CNNException
{
	using CNNException::CNNException;
};

/*
 * @brief Specifies activation function type
 */
enum class ActivationFunction
{
	Sigmoid,
	Tanh,
	ReLU,
	LeakyReLU,
	SoftMax,
	None
};

/*
 * @brief Activation layer
 */
template <class _ForwardType, class _WeightType>
class ActivationLayer : public ILayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Initializes abstract activation layer
	 *
	 * @param input     Dimensions of input matrix
	 * @param op        Activation function type
	 * 
	 */
	ActivationLayer(const Dimensions & input, const ActivationFunction & op)
		: inputSize(input)
		, outputSize(input)
		, activationFunction(op)
		, output(input)
		, gradientOutput(input)
	{
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
	 * @brief Returns type of activation function used
	 */
	ActivationFunction getActivationFunctionType() const
	{
		return activationFunction;
	}


	/*
	 * @brief Returns reference to layer output
	 */
	Image<_ForwardType> & getOutput()
	{
		return output;
	}


	/*
	 * @brief Returns reference to layer gradient output
	 */
	Image<BackwardType> & getGradientOutput()
	{
		return gradientOutput;
	}

protected:

	/// Accepted input size
	Dimensions inputSize;

	/// Output size
	Dimensions outputSize;

	/// Operation to perform
	ActivationFunction activationFunction;

	/// Output to be forward propagated to next layer
	Image<_ForwardType> output;

	/// Gradients to be backward propagated to previous layer
	Image<BackwardType> gradientOutput;

};

#endif