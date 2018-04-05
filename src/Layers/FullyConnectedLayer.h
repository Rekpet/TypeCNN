/*
 * @author Petr Rek
 * @project CNN library
 * @brief Fully connected (dense) layer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "src/Layers/ILayer.h"

#include "src/Image.h"
#include "src/Utils/Limits.h"

/*
 * @brief Exception thrown by this layer
 */
class FullyConnectedLayerException : public CNNException 
{ 
	using CNNException::CNNException; 
};

/*
 * @brief Implements fully connected layer
 */
template <class _ForwardType, class _WeightType>
class FullyConnectedLayer : public ILayer<_ForwardType, _WeightType>
{

public:

	/*
	 * @brief Creates Fully Connected Layer with given parameters and initializes it for use
	 *
	 * @param  input     Dimensions of input matrix
	 * @param  output    Dimensions of output matrix
	 * @param  useBias   Whether to use bias for each neuron
	 */
	FullyConnectedLayer(
		const Dimensions & input,
		const Dimensions & output,
		const bool useBias = true)
		: inputDimensions(input)
		, outputDimensions(output)
		, inputSize(input.depth * input.height * input.width)
		, outputSize(output.depth * output.height * output.width)
		, useBias(useBias)
		, weights(Dimensions{ inputSize + 1, outputSize, 1 })
		, deltas(Dimensions{ inputSize + 1, outputSize, 1 })
		, output(output)
		, gradientOutput(input)
	{
		// Check validity of parameters
		if (inputSize == 0 || outputSize == 0)
		{
			throw FullyConnectedLayerException("Dense layers must have at least one neuron.");
		}

		auto multiplier = computeWeightMultiplier(inputSize);

		// Initialize all weights to random value (based on number of inputs of target neuron)
		deltas.clear();
		for (auto inputNeuron = 0u; inputNeuron <= inputSize; inputNeuron++)
		{
			for (auto outputNeuron = 0u; outputNeuron < outputSize; outputNeuron++)
			{
				weights(inputNeuron, outputNeuron, 0) = generateRandomWeight(inputSize, multiplier);
			}
		}

		// Set bias to zero if it should not be used (effectively killing it)
		if (!useBias)
		{
			bias = 0.0f;
		}
	}


	/*
	 * @brief Runs the neural net on given image
	 */
	virtual void forwardPropagation(const Image<_ForwardType> & in, Image<_ForwardType> & out) override
	{
		// Check dimensions of input image
		if (in.getDimensions() != inputDimensions)
		{
			throw InputImageDoesNotHaveCorrectDimensions("Input of fully connected layer has different dimensions than declared during initilization.");
		}

		// Compute outputs
		for (auto outputNeuron = 0u; outputNeuron < outputSize; outputNeuron++)
		{
			const auto offset = outputNeuron * (inputSize + 1); // Computes offset to avoid mapping function cost (with multiplication)
			_ForwardType accum = bias * static_cast<_ForwardType>(static_cast<_WeightType>(weights(offset + inputSize)));

			for (auto inputNeuron = 0u; inputNeuron < inputSize; inputNeuron++)
			{
				accum += in(inputNeuron) * static_cast<_ForwardType>(static_cast<_WeightType>(weights(offset + inputNeuron)));
			}

			out(outputNeuron) = accum;
		}
	}


	/*
	 * @brief Runs backward propagation, learning mechanism
	 */
	virtual void backwardPropagation(const Image<_ForwardType> & input, const Image<_ForwardType> &, const Image<BackwardType> & inGradients, Image<BackwardType> & outGradients, const TrainingSettings & trainingSettings) override
	{
		outGradients.clear();

		// Compute gradients for input layer that will be propagated
		for (auto outputNeuron = 0u; outputNeuron < outputSize; outputNeuron++)
		{
			const auto offset = outputNeuron * (inputSize + 1);

			// Propagate error to previous layers
			for (auto inputNeuron = 0u; inputNeuron < inputSize; inputNeuron++)
			{
				outGradients(inputNeuron) += weights(offset + inputNeuron) * inGradients(outputNeuron);
			}
		}

		// Compute deltas
		for (auto outputNeuron = 0u; outputNeuron < outputSize; outputNeuron++)
		{
			const auto offset = outputNeuron * (inputSize + 1);

			// Update bias delta
			deltas(offset + inputSize) += static_cast<BackwardType>(bias) * inGradients(outputNeuron);

			for (auto inputNeuron = 0u; inputNeuron < inputSize; inputNeuron++)
			{
				// Update delta (for batches)
				deltas(offset + inputNeuron) += static_cast<BackwardType>(input(inputNeuron)) * inGradients(outputNeuron);
			}
		}

		// Update weights if batch size was met
		if (++examplesSinceUpdate == trainingSettings.batchSize)
		{
			this->optimizer->updateWeights(weights, deltas, examplesSinceUpdate);
			examplesSinceUpdate = 0;
		}
	}


	/*
	 * @brief Initializes the optimizer
	 */
	virtual void initializeOptimizer() override
	{
		this->optimizer->initialize(0, 0, weights.getDimensions(), 1);
	}


	/*
	 * @brief Returns a reference to layer output
	 */
	virtual Image<_ForwardType> & getOutput() override
	{
		return output;
	}


	/*
	 * @brief Returns a reference to layers output gradients
	 */
	virtual Image<BackwardType> & getGradientOutput() override
	{
		return gradientOutput;
	}


	/*
	 * @brief Returns expected input dimensions
	 */
	virtual Dimensions getInputSize() const override
	{
		return inputDimensions;
	}


	/*
	 * @brief Returns output dimensions
	 */
	virtual Dimensions getOutputSize() const override
	{
		return outputDimensions;
	}


	/*
	 * @brief Returns all weights
	 */
	Image<BackwardType> getNeuronWeights() const
	{
		return weights;
	}


	/*
	 * @brief Loads weights
	 */
	void setNeuronWeights(const Image<BackwardType> & newWeights)
	{
		if (weights.getDimensions() != newWeights.getDimensions())
		{
			throw FullyConnectedLayerException("Weights could not be loaded due to inconsistent size.");
		}

		weights = newWeights;
	}


	/*
	 * @brief Returns true if bias is used
	 */
	bool usesBias() const
	{
		return useBias;
	}

private:

	/*
	 * @brief If we are using type with just a few bits we may have as low precision at the beginning that
	 *             all weights are zeroes. We need to counter that.
	 */
	float computeWeightMultiplier(unsigned inputs)
	{
		auto epsValue = static_cast<float>(Limits::getEpsilonValue<_WeightType>());
		auto maxWeight = 1.0f / sqrt(inputs) / 1.25f;
		auto multiplier = 1.0f;

		while ((maxWeight * multiplier) < epsValue)
		{
			multiplier += 1.0f;
		}

		return multiplier;
	}

	/*
	 * @brief Generates random weight based on number of inputs of neuron
	 *        <-1/sqrt(inputs), 1/sqrt(inputs)>
	 */
	BackwardType generateRandomWeight(unsigned inputs, float multiplier = 1.0f) const
	{
		auto randomVal = static_cast<BackwardType>((static_cast<float>(rand()) / RAND_MAX) * 2 - 1);
		return randomVal * (static_cast<BackwardType>(1.0f * multiplier) / static_cast<BackwardType>(static_cast<float>(sqrt(inputs))));
	}

private:

	/// Accepted input size
	Dimensions inputDimensions;

	/// Output size
	Dimensions outputDimensions;

	/// Number of neurons in input layer
	unsigned inputSize;

	/// Number of neurons in output layer
	unsigned outputSize;

	/// Bias value
	_ForwardType bias = 1.0f;

	/// Should bias be used
	bool useBias;

	/// Number of training examples since updating weights
	unsigned examplesSinceUpdate = 0;

	/// Weights between neurons
	Image<BackwardType> weights;

	/// Deltas for updating weights (needed for batches)
	Image<BackwardType> deltas;

	/// Output to be forward propagated to next layer
	Image<_ForwardType> output;

	/// Gradients to be backward propagated to previous layer
	Image<BackwardType> gradientOutput;


};

#endif