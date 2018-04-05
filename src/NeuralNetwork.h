/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Neural network instance
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#ifdef _MSC_VER
#pragma once
#endif

#include "ConvolutionalNeuralNetwork.h"
#include "Image.h"

#include "Layers/ActivationLayer.h"

/*
 * @brief An instance of Neural Network, both for usage and training
 */
class NeuralNetwork : public ConvolutionalNeuralNetwork
{

private:

	using ConvolutionalNeuralNetwork::addLayer; // Limited interface for NN
	using ConvolutionalNeuralNetwork::run;
	using ConvolutionalNeuralNetwork::train;
	using ConvolutionalNeuralNetwork::validate;

public:

	NeuralNetwork(unsigned inputLayerSize,
		std::vector<std::pair<unsigned, ActivationFunction>> hiddenLayers,
		std::pair<unsigned, ActivationFunction> outputLayer,
		const bool useBias = true,
		const TaskType & taskType = TaskType::Classification);

	Image<ForwardType> run(const std::vector<ForwardType> & input);

	float train(TrainingSettings & settings, 
		const std::vector<std::pair<std::vector<ForwardType>, std::vector<ForwardType>>> & trainingData, 
		const LossFunctionType & lossFunction, const std::shared_ptr<IOptimizer> optimizer, 
		const std::vector<std::pair<std::vector<ForwardType>, std::vector<ForwardType>>> & validationData = {});

	float validate(const std::vector<std::pair<std::vector<ForwardType>, std::vector<ForwardType>>> & data);

};

#endif