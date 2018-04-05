/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Convolutional neural network instance
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef CONVOLUTIONAL_NEURAL_NETWORK_H
#define CONVOLUTIONAL_NEURAL_NETWORK_H

#include "src/Layers/ILayer.h"
#include "src/CompileSettings.h"
#include "src/TrainingSettings.h"
#include "src/Image.h"
#include "src/LayerAliases.h"

#include <functional>
#include <utility>

// epoch num, training settings, epoch error, validation accuracy, epoch length
using OnEpochFinishedCallbackType = std::function<void(unsigned, TrainingSettings &, float, float, float)>;

/*
 * @brief An instance of Convolutional Neural Network, both for usage and training
 */
class ConvolutionalNeuralNetwork
{

public:

	ConvolutionalNeuralNetwork(const TaskType & taskType = TaskType::Classification);

	void addLayer(const std::shared_ptr<ILayer<ForwardType, WeightType>> layer);

	Image<ForwardType> run(const Image<ForwardType> & input);

	float train(TrainingSettings & settings, std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & trainingData, 
		const LossFunctionType & lossFunction, const std::shared_ptr<IOptimizer> optimizer,
		const std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & validationData = {});

	float validate(const std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & data);

	void setOnEpochFinishedCallback(OnEpochFinishedCallbackType callback);

	void enableOutput();

	void disableOutput();

	TaskType getTaskType() const;

	Dimensions getInputSize() const;

	Dimensions getOutputSize() const;

public:

	/// Layer iterator
	std::vector<std::shared_ptr<ILayer<ForwardType, WeightType>>>::const_iterator begin() const 
	{ 
		return allLayers.begin(); 
	}

	/// Layer iterator
	std::vector<std::shared_ptr<ILayer<ForwardType, WeightType>>>::const_iterator end() const 
	{ 
		return allLayers.end(); 
	}

private:

	void printResults(const Image<ForwardType> & output) const;

	std::pair<BackwardType, Image<BackwardType>> computeError(const Image<ForwardType> & actual, 
		const Image<ForwardType> & expected, const LossFunctionType & lossFunctionType) const;

private:

	/// Layers not used during training
	std::vector<std::shared_ptr<ILayer<ForwardType, WeightType>>> forwardOnlyLayers;

	/// Layers used both in training and inference
	std::vector<std::shared_ptr<ILayer<ForwardType, WeightType>>> allLayers;

	/// Number of layers used only during inference
	unsigned forwardOnlyLayerNum = 0;

	/// Number of layers used both when training and during inference
	unsigned allLayerNum = 0;

	/// Task type
	TaskType taskType = TaskType::Classification;

	/// Accepted input size
	Dimensions inputSize = { 0, 0, 0 };

	/// Output size
	Dimensions outputSize = { 0, 0, 0 };

	/// Output to std::cout is enabled
	bool outputEnabled = false;

	/// Suppresses output during training
	bool suppressOutput = false;

	/// Specifies that training is in progress
	bool training = false;

	/// Function to call when epoch finishes
	OnEpochFinishedCallbackType onEpochFinishedCallback = nullptr;

};

#endif