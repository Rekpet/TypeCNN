/*
* @author Petr Rek
* @project CNN Library
* @brief Conversion layer to convert different fixed point types
*/

#ifdef _MSC_VER
#pragma once
#endif

#ifndef CONVERSION_LAYER_H
#define CONVERSION_LAYER_H

#include "src/Image.h"
#include "src/TrainingSettings.h"

/*
 * @brief Special conversion layer to convert I/O between fixed point layers
 */
template <class PrevLayerType, class NextLayerType>
class ConversionLayer
{

public:

	/*
	 * @brief Sets up matrices to hold output values
	 *
	 * @param size    Dimensions of both input and output matrix
	 */
	ConversionLayer(const Dimensions & size)
		: size(size)
		, output(size)
		, gradientOutput(size)
	{
	}

	/*
	 * @brief Converts between different FixedPoint types
	 */
	void forwardPropagation(const Image<PrevLayerType> & in, Image<NextLayerType> & out)
	{
		auto flattenedSize = in.getFlattenedSize();

		for (auto i = 0u; i < flattenedSize; i++)
		{
			out(i) = static_cast<NextLayerType>(static_cast<float>(in(i)));
		}
	}


	/*
	 * @brief Backpropagation always uses same types
 	 */
	void backwardPropagation(const Image<PrevLayerType> &, const Image<NextLayerType> &, const Image<BackwardType> & inGradients, Image<BackwardType> & outGradients, const TrainingSettings &)
	{
		outGradients = inGradients;
	}


	/*
	 * @brief Returns a reference to layer output
	 */
	Image<NextLayerType> & getOutput()
	{
		return output;
	}


	/*
	 * @brief Returns a reference to layer gradient output
	 */
	Image<BackwardType> & getGradientOutput()
	{
		return gradientOutput;
	}

	/*
	 * @brief Returns expected input dimensions
	 */
	Dimensions getInputSize() const
	{
		return size;
	}


	/*
	 * @brief Returns output dimensions
	 */
	Dimensions getOutputSize() const
	{
		return size;
	}

private:

	/// Input and output size
	Dimensions size;

	/// Output to be forward propagated to next layer
	Image<NextLayerType> output;

	/// Gradients to be backward propagated to previous layer
	Image<BackwardType> gradientOutput;

};

#endif