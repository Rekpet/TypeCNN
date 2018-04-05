/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Maximum pooling layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H

#include "src/Layers/ILayer.h"
#include "src/Layers/PoolingLayer.h"

#include "src/Image.h"
#include "src/Utils/Limits.h"

/*
 * @brief Max pooling layer that reduces width and height of input matrix
 */
template <class _ForwardType, class _WeightType>
class MaxPoolingLayer : public PoolingLayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Checks parameters and validates output size
	 */
	MaxPoolingLayer(const Dimensions & input, const unsigned & extent, const unsigned & stride)
		: PoolingLayer<_ForwardType, _WeightType>(input, extent, stride, PoolingOperation::Max)
	{
	}


	/*
	 * @brief Forward propagates a matrix in order to reduce dimension using given operation
	 */
	virtual void forwardPropagation(const Image<_ForwardType> & in, Image<_ForwardType> & out) override
	{
		if (in.getDimensions() != this->inputSize)
		{
			throw InputImageDoesNotHaveCorrectDimensions("Input image does not correspond to declared input size in Pooling layer.");
		}

		auto flattenedSize = out.getFlattenedSize();

		// Perform pooling
		_ForwardType initAccumValue = Limits::getMinimumValue<_ForwardType>();
		for (auto i = 0u; i < flattenedSize; i++)
		{
			auto accum = initAccumValue;
			for (auto k = 0u; k < this->windowSize; k++)
			{
				if (in(this->edges[i][k]) > accum)
				{
					accum = in(this->edges[i][k]);
				}
			}
			out(i) = accum;
		}

		// Slower, but more descriptive implementation for future reference
		/*for (auto z = 0u; z < outputSize.depth; z++)
		{
			unsigned currY = 0;
			for (auto i = 0u; i < outputSize.height; i++)
			{
				unsigned currX = 0;
				for (auto j = 0u; j < outputSize.width; j++)
				{
					auto accum = initAccumValue;
					for (auto b = 0u; b < extent; b++)
					{
						for (auto a = 0u; a < extent; a++)
						{
							unsigned x = currX + a;
							unsigned y = currY + b;

							if (in(x, y, z) > accum)
							{
								accum = in(x, y, z);
							}
						}
					}
					out(j, i, z) = accum;
					currX += stride;
				}
				currY += stride;
			}
		}*/
	}


	/*
	 * @brief Computes gradients to be propagated to previous layer (depends on usder operation)
	 */
	virtual void backwardPropagation(const Image<_ForwardType> & in, const Image<_ForwardType> & out, const Image<BackwardType> & inGradients, Image<BackwardType> & outGradients, const TrainingSettings &) override
	{
		outGradients.clear();

		auto flattenedSize = inGradients.getFlattenedSize();

		// Reverse pooling (assign error to min/max element or split it if average was used)
		for (auto i = 0u; i < flattenedSize; i++)
		{
			for (auto k = 0u; k < this->windowSize; k++)
			{
				if (in(this->edges[i][k]) == out(i))
				{
					outGradients(this->edges[i][k]) += inGradients(i);
				}
			}
		}

		// Slower, but more descriptive implementation for future reference
		/*for (auto z = 0u; z < outputSize.depth; z++)
		{
			unsigned currY = 0;
			for (auto i = 0u; i < outputSize.height; i++)
			{
				unsigned currX = 0;
				for (auto j = 0u; j < outputSize.width; j++)
				{
					for (auto b = 0u; b < extent; b++)
					{
						for (auto a = 0u; a < extent; a++)
						{
							unsigned x = currX + a;
							unsigned y = currY + b;
							if (in(x, y, z) == out(j, i, z))
							{
								outGradients(x, y, z) += inGradients(j, i, z);
							}
						}
					}
					currX += stride;
				}
				currY += stride;
			}
		}*/
	}

};

#endif