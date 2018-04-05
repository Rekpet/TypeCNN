/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Universal pooling layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "src/Layers/ILayer.h"

#include "src/Image.h"

#include <limits>

/*
 * @brief Exception thrown if problems occur during PoolingLayer constructor
 */
class PoolingLayerException : public CNNException
{
	using CNNException::CNNException;
};

/*
 * @brief Type of pooling operation to be executed
 */
enum class PoolingOperation
{
	Max,
	Average
};

/*
 * @brief Pooling layer that reduces width and height of input matrix
 */
template <class _ForwardType, class _WeightType>
class PoolingLayer : public ILayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Checks parameters and validates output size
	 */
	PoolingLayer(const Dimensions & input, const unsigned & extent, const unsigned & stride, const PoolingOperation & op)
		: inputSize(input)
		, extent(extent)
		, stride(stride)
		, operation(op)
		, gradientOutput(input)
	{
		if (stride == 0 || extent == 0)
		{
			throw PoolingLayerException("Stride or window size set to zero.");
		}

		// Calculate new dimension sizes
		auto newWidth = (inputSize.width - extent) / static_cast<float>(stride) + 1;
		auto newHeight = (inputSize.height - extent) / static_cast<float>(stride) + 1;

		outputSize.width = static_cast<unsigned>(newWidth);
		outputSize.height = static_cast<unsigned>(newHeight);
		outputSize.depth = inputSize.depth;
		windowSize = extent * extent;

		// Check that filter can be applied:
		//     - output dimensions are whole numbers
		//     - input is not smaller than filters
		if ((fabs(newWidth - outputSize.width) > 0.0001f)
			|| (fabs(newHeight - outputSize.height) > 0.0001f)
			|| (inputSize.width < extent)
			|| (inputSize.height < extent))
		{
			throw PoolingLayerException("Cannot apply pooling of these parameters on declared input size.");
		}

		// Initialize output matrix
		output = Image<_ForwardType>(outputSize);
		edges.resize(outputSize.width * outputSize.height * outputSize.depth);

		createEdges();
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
	 * @brief Returns pooling operation type
	 */
	PoolingOperation getPoolingOperationType() const
	{
		return operation;
	}


	/*
	 * @brief Returns extent size
	 */
	unsigned getExtentSize() const
	{
		return extent;
	}


	/*
	 * @brief Returns stride
	 */
	unsigned getStride() const
	{
		return stride;
	}


	/*
	 * @brief Returns reference to output matrix
	 */
	virtual Image<_ForwardType> & getOutput() override
	{
		return output;
	}


	/*
	 * @brief Returns reference to gradient output
	 */
	virtual Image<BackwardType> & getGradientOutput() override
	{
		return gradientOutput;
	}

protected:

	/*
	* @brief Creates edges to make pooling operation faster
	*/
	void createEdges()
	{
		for (auto z = 0u; z < outputSize.depth; z++)
		{
			unsigned currY = 0;
			for (auto i = 0u; i < outputSize.height; i++)
			{
				unsigned currX = 0;
				for (auto j = 0u; j < outputSize.width; j++)
				{
					auto index = z * outputSize.height * outputSize.width + i * outputSize.width + j;
					edges[index].resize(windowSize);
					auto cnt = 0u;
					for (auto b = 0u; b < extent; b++)
					{
						for (auto a = 0u; a < extent; a++)
						{
							unsigned x = currX + a;
							unsigned y = currY + b;
							edges[index][cnt++] = z * inputSize.width * inputSize.height + y * inputSize.width + x;
						}
					}
					currX += stride;
				}
				currY += stride;
			}
		}
	}

protected:
	
	/// Holds edges from each output point to input points
	std::vector<std::vector<unsigned>> edges;

	/// Accepted input size
	Dimensions inputSize;

	/// Output size
	Dimensions outputSize;

	/// Size of window
	unsigned extent;

	/// Stride
	unsigned stride;

	/// Operation to perform
	PoolingOperation operation;

	/// Window size
	unsigned windowSize;

	/// Output to be forward propagated to next layer
	Image<_ForwardType> output;

	/// Gradients to be backward propagated to previous layer
	Image<BackwardType> gradientOutput;

};

#endif
