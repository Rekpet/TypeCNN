/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Neural network instance
 */

#include "src/NeuralNetwork.h"

#include "src/Layers/FullyConnectedLayer.h"
#include "src/Utils/PersistenceMapper.h"

#include <algorithm>
#include <iostream>
#include <iomanip>

/*
 * @brief Constructor, sets up layers of neural network
 *
 * @param inputLayerSize      Number of input neurons (inputs)
 * @param hiddenLayers        Vector of pairs (number of neurons, activation function)
 * @param outputLayer         Pair (number of output neurons, activation function)
 * @param useBias             Whether to use biases
 * @param taskType            Task type to be performed (classification | regression)
 */
NeuralNetwork::NeuralNetwork(unsigned inputLayerSize,
	std::vector<std::pair<unsigned, ActivationFunction>> hiddenLayers,
	std::pair<unsigned, ActivationFunction> outputLayer,
	const bool useBias /*= true*/,
	const TaskType & taskType /*= TaskType::Classification*/)
	: ConvolutionalNeuralNetwork(taskType)
{
	hiddenLayers.push_back(outputLayer);
	auto prevLayerSize = inputLayerSize;
	for (const auto & layer : hiddenLayers)
	{		
		addLayer(std::make_shared<FullyConnectedLayer<ForwardType, WeightType>>(Dimensions{ prevLayerSize, 1, 1 }, Dimensions{ layer.first, 1, 1 }, useBias));
		prevLayerSize = layer.first;

		if (layer.second != ActivationFunction::None)
		{
			addLayer(PersistenceMapper::getActivationLayer(layer.second, Dimensions{ layer.first, 1, 1 }));
		}
	}
}


/*
 * @brief Propagates an input throught neural network
 *
 * @param  input     Matrix with input data
 *
 * @return output    Output matrix
 */
Image<ForwardType> NeuralNetwork::run(const std::vector<ForwardType> & input)
{
	return ConvolutionalNeuralNetwork::run(Image<ForwardType>(input));
}


/*
 * @brief Trains a neural network on set of examples
 *
 * @param  settings      Training settings for learning (batch size, number of epochs..)
 * @param  data          Vector of pairs (input, expected output)
 * @param  lossFunction  Loss function to be used
 * @param  optimizer     Optimizer to be used
 *
 * @return error         Error is last epoch
 * @throws CNNException if no data were passed for training
 */
float NeuralNetwork::train(TrainingSettings & settings, 
	const std::vector<std::pair<std::vector<ForwardType>, std::vector<ForwardType>>> & trainingData,
	const LossFunctionType & lossFunction, const std::shared_ptr<IOptimizer> optimizer, 
	const std::vector<std::pair<std::vector<ForwardType>, std::vector<ForwardType>>> & validationData /*= {}*/)
{
	std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> inputTrainingData;
	for (const auto & input : trainingData)
	{
		inputTrainingData.emplace_back(Image<ForwardType>{input.first}, Image<ForwardType>{input.second});
	}

	std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> inputValidatioData;
	for (const auto & input : validationData)
	{
		inputValidatioData.emplace_back(Image<ForwardType>{ input.first }, Image<ForwardType>{ input.second });
	}

	return ConvolutionalNeuralNetwork::train(settings, inputTrainingData, lossFunction, optimizer, inputValidatioData);
}


/*
 * @brief Validates a network on test dataset
 *
 * @param  data       Input data for training
 *
 * @return accuracy   Average error for regression, accuraccy for classification
 * @throws CNNException if no data were passed for validation
 */
float NeuralNetwork::validate(const std::vector<std::pair<std::vector<ForwardType>, std::vector<ForwardType>>> & data)
{
	std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> inputData;

	for (const auto & input : data)
	{
		inputData.emplace_back(Image<ForwardType>{ input.first }, Image<ForwardType>{ input.second });
	}

	return ConvolutionalNeuralNetwork::validate(inputData);
}