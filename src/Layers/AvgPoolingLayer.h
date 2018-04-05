/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Average pooling layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef AVG_POOLING_LAYER_H
#define AVG_POOLING_LAYER_H

#include "src/Layers/ILayer.h"
#include "src/Layers/PoolingLayer.h"

#include "src/Image.h"

/*
 * @brief Average pooling layer that reduces width and height of input matrix
 */
template <class _ForwardType, class _WeightType>
class AvgPoolingLayer : public PoolingLayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Initialized average pooling layer
	 *
	 * @param input      Dimensions of input matrix
	 * @param extent     Extent of 2D window used for pooling
	 * @param stride     Step with which the window is slided
	 */
	AvgPoolingLayer(const Dimensions & input, const unsigned & extent, const unsigned & stride)
		: PoolingLayer<_ForwardType, _WeightType>(input, extent, stride, PoolingOperation::Average)
	{
	}


	/*
	 * @brief Forward propagates a matrix in order to reduce dimension using given operation
	 */
	virtual void forwardPropagation(const Image<_ForwardType> & in, Image<_ForwardType> & out) override
	{
		if (in.getDimensions() != this->inputSize)
		{
			throw InputImageDoesNotHaveCorrectDimensions("Input image does not correspond to "
				"declared input size in Pooling layer.");
		}

		auto flattenedSize = out.getFlattenedSize();

		// Perform pooling
		for (auto i = 0u; i < flattenedSize; i++)
		{
			auto accum = static_cast<_ForwardType>(0);
			for (auto k = 0u; k < this->windowSize; k++)
			{
				accum += in(this->edges[i][k]);
			}
			out(i) = accum / static_cast<_ForwardType>(static_cast<float>(this->windowSize));
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
							accum += in(x, y, z);
						}
					}
					out(j, i, z) = accum / static_cast<_ForwardType>(static_cast<float>(extent * extent));
					currX += stride;
				}
				currY += stride;
			}
		}*/
	}


	/*
	 * @brief Computes gradients to be propagated to previous layer (depends on usder operation)
	 */
	virtual void backwardPropagation(const Image<_ForwardType> &, const Image<_ForwardType> &, 
		const Image<BackwardType> & inGradients, Image<BackwardType> & outGradients, const TrainingSettings &) override
	{
		outGradients.clear();

		auto flattenedSize = inGradients.getFlattenedSize();

		// Reverse pooling (assign error to min/max element or split it if average was used)
		for (auto i = 0u; i < flattenedSize; i++)
		{
			for (auto k = 0u; k < this->windowSize; k++)
			{
				outGradients(this->edges[i][k]) += inGradients(i) / static_cast<BackwardType>(static_cast<float>(this->windowSize));
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
							outGradients(x, y, z) += inGradients(j, i, z) / (extent * extent);
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