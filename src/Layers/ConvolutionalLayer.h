/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Universal convolutional layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "src/Layers/ILayer.h"

#include "src/Image.h"
#include "src/Utils/Limits.h"

#include <algorithm>

/*
 * @brief Exception thrown if problems occur during initialization
 */
class ConvolutionalLayerException : public CNNException 
{ 
	using CNNException::CNNException;
};

/*
 * @brief Convolutional layer
 */
template <class _ForwardType, class _WeightType>
class ConvolutionalLayer : public ILayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Creates Convolutional Layer with given parameters and initializes it for use
	 *
	 * @param input          Dimensions of input matrix
	 * @param stride         Stride with which filter is slided across input
	 * @param filterNum      Number of 3D filters
	 * @param filterExtent   Height and width of filters
	 * @param zeroPadding    Zero padding to be put around input
	 * @param useBias        Whether to use biases for each filter
	 */
	ConvolutionalLayer(
		const Dimensions & input
		, const unsigned stride
		, const unsigned filterNum
		, const unsigned filterExtent
		, const unsigned zeroPadding = 0
		, const bool useBias = true)
		: inputSize(input)
		, useBias(useBias)
		, filterNum(filterNum)
		, filterExtent(filterExtent)
		, stride(stride)
		, zeroPadding(zeroPadding)
		, gradientOutput(input)
	{
		// Check stride
		if (stride == 0 || filterNum == 0 || filterExtent == 0)
		{
			throw ConvolutionalLayerException("Stride, filter extent or filter number were set to zero.");
		}

		auto multiplier = computeWeightMultiplier();

		// Create empty filters and biases
		for (auto i = 0u; i < filterNum; i++)
		{
			filters.push_back(Image<BackwardType>(Dimensions{ filterExtent, filterExtent, input.depth }));
			filterDeltas.push_back(Image<BackwardType>(Dimensions{ filterExtent, filterExtent, input.depth }));
			filterDeltas.back().clear();

			biases.push_back(generateRandomWeight(multiplier));
			biasDeltas.push_back(0);
		}

		// Calculate output dimension and check if all parameters work together
		auto newHeight = (static_cast<int>(inputSize.height) - filterExtent + 2 * zeroPadding) / static_cast<float>(stride) + 1;
		auto newWidth = (static_cast<int>(inputSize.width) - filterExtent + 2 * zeroPadding) / static_cast<float>(stride) + 1;

		outputSize.width = static_cast<unsigned>(newWidth);
		outputSize.height = static_cast<unsigned>(newHeight);
		outputSize.depth = filterNum;
		windowSize = filterExtent * filterExtent * inputSize.depth;

		// Check that convolution can be applied
		if ((fabs(newWidth - outputSize.width) > 0.0001f)
			|| (fabs(newHeight - outputSize.height) > 0.0001f)
			|| ((inputSize.width + 2 * zeroPadding) < filterExtent)
			|| ((inputSize.height + 2 * zeroPadding) < filterExtent))
		{
			throw ConvolutionalLayerException("Convolutions settings cannot be applied on given input dimensions.");
		}

		// Initialize output matrix
		output = Image<_ForwardType>(outputSize);
		inputEdges.resize(outputSize.width * outputSize.height);
		filterEdges.resize(outputSize.width * outputSize.height);

		// Initialize filters with random values
		for (auto f = 0u; f < filterNum; f++)
		{
			for (auto i = 0u; i < input.depth; i++)
			{
				for (auto j = 0u; j < filterExtent; j++)
				{
					for (auto k = 0u; k < filterExtent; k++)
					{
						filters[f](k, j, i) = generateRandomWeight(multiplier);
					}
				}
			}
		}

		createEdges();
	}


	/*
	 * @brief Propagates the image through convolutional layer obtainign features on output
	 */
	virtual void forwardPropagation(const Image<_ForwardType> & in, Image<_ForwardType> & out) override
	{
		if (in.getDimensions() != inputSize)
		{
			throw InputImageDoesNotHaveCorrectDimensions("Input to convolutional layer has different dimensions than declared.");
		}

		// Slides 3D filter accross matrix and computes output values
		auto flattenedSize = outputSize.width * outputSize.height;

		// If there is no padding, we may optimize the code
		if (zeroPadding == 0)
		{
			for (auto filter = 0u; filter < filterNum; filter++)
			{
				const auto offset = filter * flattenedSize;
				const auto initAccumValue = (useBias)
												? (static_cast<_ForwardType>(static_cast<_WeightType>(biases[filter])))
												: (static_cast<_ForwardType>(0.0f));

				for (auto i = 0u; i < flattenedSize; i++)
				{
					auto accum = initAccumValue;

					for (auto k = 0u; k < windowSize; k++)
					{
						accum += in(inputEdges[i][k]) * static_cast<_ForwardType>(static_cast<_WeightType>(filters[filter](filterEdges[i][k])));
					}

					out(i + offset) = accum;
				}
			}
		}
		else
		{
			for (auto filter = 0u; filter < filterNum; filter++)
			{
				const auto offset = filter * flattenedSize;
				const auto initAccumValue = (useBias)
												? (static_cast<_ForwardType>(static_cast<_WeightType>(biases[filter])))
												: (static_cast<_ForwardType>(0.0f));

				for (auto i = 0u; i < flattenedSize; i++)
				{
					auto accum = initAccumValue;

					for (auto k = 0u; k < windowSize; k++)
					{
						if (inputEdges[i][k] >= 0)
						{
							accum += in(inputEdges[i][k]) * static_cast<_ForwardType>(static_cast<_WeightType>(filters[filter](filterEdges[i][k])));
						}
					}

					out(i + offset) = accum;
				}
			}
		}

		// Slower, but more descriptive implementation for future reference
		/*auto currentWidth = static_cast<int>(inputSize.width);
		auto currentHeight = static_cast<int>(inputSize.height);
		for (auto filter = 0u; filter < filterNum; filter++)
		{
			int filterY = -static_cast<int>(zeroPadding);

			for (auto j = 0u; j < outputSize.height; j++)
			{
				int filterX = -static_cast<int>(zeroPadding);

				for (auto k = 0u; k < outputSize.width; k++)
				{
					auto accum = static_cast<_ForwardType>(biases[filter]);

					for (auto i = 0u; i < in.getDepth(); i++)
					{
						for (auto b = 0u; b < filterExtent; b++)
						{
							for (auto a = 0u; a < filterExtent; a++)
							{
								int x = filterX + a;
								int y = filterY + b;

								if (x >= 0 && y >= 0 && x < currentWidth && y < currentHeight)
								{
									accum += in(x, y, i) * static_cast<_ForwardType>(filters[filter](a, b, i));
								}
							}
						}
					}

					filterX += stride;
					out(k, j, filter) = accum;
				}

				filterY += stride;
			}
		}*/
	}


	/*
	 * @brief Updated value of filters and biases, computes gradients for previous layer
	 */
	virtual void backwardPropagation(const Image<_ForwardType> & in, const Image<_ForwardType> &, const Image<BackwardType> & inGradients, 
		Image<BackwardType> & outGradients, const TrainingSettings & trainingSettings) override
	{
		outGradients.clear();

		// Reverses operation to compute how each input contributed to overall error
		auto flattenedSize = outputSize.width * outputSize.height;

		// If there is no padding, we may optimize the code
		if (zeroPadding == 0)
		{	
			for (auto filter = 0u; filter < filterNum; filter++)
			{
				const auto offset = filter * flattenedSize;

				for (auto i = 0u; i < flattenedSize; i++)
				{
					const auto index = i + offset;
					biasDeltas[filter] += inGradients(index);

					for (auto k = 0u; k < windowSize; k++)
					{
						outGradients(inputEdges[i][k]) += filters[filter](filterEdges[i][k]) * inGradients(index);
						filterDeltas[filter](filterEdges[i][k]) += inGradients(index) * static_cast<BackwardType>(in(inputEdges[i][k]));
					}
				}
			}
		}
		else
		{
			for (auto filter = 0u; filter < filterNum; filter++)
			{
				const auto offset = filter * flattenedSize;

				for (auto i = 0u; i < flattenedSize; i++)
				{
					const auto index = i + offset;
					biasDeltas[filter] += inGradients(index);

					for (auto k = 0u; k < windowSize; k++)
					{
						if (inputEdges[i][k] >= 0)
						{
							outGradients(inputEdges[i][k]) += filters[filter](filterEdges[i][k]) * inGradients(index);
							filterDeltas[filter](filterEdges[i][k]) += inGradients(index) * static_cast<BackwardType>(in(inputEdges[i][k]));
						}
					}
				}
			}
		}

		// Slower, but more descriptive implementation for future reference
		/*for (auto k = 0u; k < filterNum; k++)
		{
			for (auto x = 0u; x < in.getWidth(); x++)
			{
				for (auto y = 0u; y < in.getHeight(); y++)
				{
					// Find output coordinates that this input pixel affected
					int Xmin = std::max(0, static_cast<int>((static_cast<int>(x) - filterExtent + 1) / stride));
					int Xmax = x / stride;
					int Ymin = std::max(0, static_cast<int>((static_cast<int>(y) - filterExtent + 1) / stride));
					int Ymax = y / stride;

					for (auto z = 0u; z < in.getDepth(); z++)
					{
						for (auto i = Xmin; i <= Xmax; i++)
						{
							int minx = i * stride;
							for (int j = Ymin; j <= Ymax; j++)
							{
								int miny = j * stride;

								outGradients(x, y, z) += filters[k](x - minx, y - miny, z) * inGradients(i, j, k);

								filterDeltas[k](x - minx, y - miny, z) += inGradients(i, j, k) * static_cast<BackwardType>(in(x, y, z));
								biasDeltas[k] += inGradients(i, j, k) / (filterExtent * filterExtent);
							}
						}
					}
				}
			}
		}*/

		// Update weights one batch size is met
		if (++examplesSinceUpdate == trainingSettings.batchSize)
		{
			this->optimizer->updateWeights(filters, filterDeltas, examplesSinceUpdate);
			this->optimizer->updateWeights(biases, biasDeltas, examplesSinceUpdate);
			examplesSinceUpdate = 0;
		}
	}


	/*
	 * @brief Initializes the optimizer
	 */
	virtual void initializeOptimizer() override
	{
		this->optimizer->initialize(filterNum, 1, filters[0].getDimensions(), filterNum);
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
	 * @brief Returns stride
	 */
	unsigned getStride() const
	{
		return stride;
	}


	/*
	 * @brief Returns filter extent
	 */
	unsigned getExtent() const
	{
		return filterExtent;
	}


	/*
	 * @brief Returns number of filters (and feature maps)
	 */
	unsigned getFilterNum() const
	{
		return filterNum;
	}


	/*
	 * @brief Returns if bias is used
	 */
	bool usesBias() const
	{
		return useBias;
	}


	/*
	 * @brief Returns amount of zero padding used
	 */
	unsigned getZeroPadding() const
	{
		return zeroPadding;
	}


	/*
	 * @brief Returns filter values
	 */
	std::vector<Image<BackwardType>> getFilters() const
	{
		return filters;
	}


	/*
	 * @brief Returns bias values
	 */
	std::vector<BackwardType> getBiases() const
	{
		return biases;
	}


	/*
	 * @brief Loads filter and bias values
	 */
	void loadFilters(const std::vector<Image<BackwardType>> & fs, const std::vector<BackwardType> & b = {})
	{
		if (fs.size() != filterNum || (useBias && b.size() != filterNum))
		{
			throw ConvolutionalLayerException("Cannot load filters due to inconsistent amounts of filters and/or biases.");
		}

		for (const auto & f : fs)
		{
			if (f.getHeight() != filterExtent || f.getWidth() != filterExtent || f.getDepth() != inputSize.depth)
			{
				throw ConvolutionalLayerException("Cannot load filters due to inconsistent dimensions of filters.");
			}
		}

		biases = b;

		filters = fs;
	}

private:

	/*
	 * @brief Creates edges to make convolution operation faster
	 */
	void createEdges()
	{
		auto currentWidth = static_cast<int>(inputSize.width);
		auto currentHeight = static_cast<int>(inputSize.height);

		for (auto filter = 0u; filter < filterNum; filter++)
		{
			int filterY = -static_cast<int>(zeroPadding);
			for (auto j = 0u; j < outputSize.height; j++)
			{
				int filterX = -static_cast<int>(zeroPadding);
				for (auto k = 0u; k < outputSize.width; k++)
				{
					for (auto i = 0u; i < inputSize.depth; i++)
					{
						for (auto b = 0u; b < filterExtent; b++)
						{
							for (auto a = 0u; a < filterExtent; a++)
							{
								int x = filterX + a;
								int y = filterY + b;

								if (x >= 0 && y >= 0 && x < currentWidth && y < currentHeight)
								{
									inputEdges[k + j * outputSize.width].push_back(i * inputSize.height * inputSize.width + y * inputSize.width + x);
								}
								else
								{
									inputEdges[k + j * outputSize.width].push_back(-1);
								}

								filterEdges[k + j * outputSize.width].push_back(i * filterExtent * filterExtent + b * filterExtent + a);
							}
						}
					}
					filterX += stride;
				}
				filterY += stride;
			}
		}
	}


	/*
	 * @brief If we are using type with just a few bits we may have as low precision at the beginning that
	 *             all weights are zeroes. We need to counter that.
	 */
	float computeWeightMultiplier()
	{
		auto epsValue = static_cast<float>(Limits::getEpsilonValue<_WeightType>());
		auto maxWeight = (1.0f / (filterExtent * filterExtent * inputSize.depth)) / 1.25f;
		auto multiplier = 1.0f;

		while ((maxWeight * multiplier) < epsValue)
		{
			multiplier += 1.0f;
		}

		return multiplier;
	}

	/*
	 * @brief Generates random weight
	 */
	BackwardType generateRandomWeight(float multiplier = 1.0f) const
	{
		auto randomVal = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
		return static_cast<BackwardType>(multiplier * randomVal * (1.0f / (filterExtent * filterExtent * inputSize.depth)));
	}

protected:

	/// Edges used to speed up convolution
	std::vector<std::vector<int>> inputEdges;
	std::vector<std::vector<int>> filterEdges;

	/// 3D size of filter
	unsigned windowSize;

	/// Accepted input size
	Dimensions inputSize;

	/// Output size
	Dimensions outputSize;

	/// Whether to use bias
	bool useBias = false;

	/// Number of filters
	unsigned filterNum;

	/// Size of filters
	unsigned filterExtent;

	/// Stride
	unsigned stride;

	/// Number of zero borders added aroung the image
	unsigned zeroPadding;

	/// Number of examples since last update of filters/biases
	unsigned examplesSinceUpdate = 0;

	/// Filters
	std::vector<Image<BackwardType>> filters;

	/// Filter deltas
	std::vector<Image<BackwardType>> filterDeltas;

	/// Biases
	std::vector<BackwardType> biases;

	/// Bias deltas
	std::vector<BackwardType> biasDeltas;

	/// Output to be forward propagated to next layer
	Image<_ForwardType> output;

	/// Gradients to be backward propagated to previous layer
	Image<BackwardType> gradientOutput;


};

#endif
