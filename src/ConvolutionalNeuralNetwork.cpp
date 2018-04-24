/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Convolutional neural network instance
 */

#include "src/ConvolutionalNeuralNetwork.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

/*
 * @brief Constructor, initializes task type
 *
 * @param taskType    Type of task to perform (classification|regression)
 */
ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(const TaskType & taskType /*= TaskType::Classification*/)
	: taskType(taskType)
{
}


/*
 * @brief Adds layer with ILayer interface
 *
 * @param layer   Pointer to layer to be added
 */
void ConvolutionalNeuralNetwork::addLayer(const std::shared_ptr<ILayer<ForwardType, WeightType>> layer)
{
	if (allLayers.empty())
	{
		inputSize = layer->getInputSize();
	}

	if (!layer->useOnlyWhenLearning)
	{
		forwardOnlyLayers.push_back(layer);
		forwardOnlyLayerNum++;
	}

	allLayers.push_back(layer);
	allLayerNum++;

	outputSize = layer->getOutputSize();
}


/*
 * @brief Runs the Convolutional Neural Network on single image
 *
 * @param  input    Input matrix with data
 *
 * @return output   Matrix with output
 * @throws CNNException if no layers were added
 */
Image<ForwardType> ConvolutionalNeuralNetwork::run(const Image<ForwardType> & input)
{
	if (forwardOnlyLayers.empty())
	{
		throw CNNException("No layers to perform inference on.");
	}

	// Run image through all layers, emits layers used only for training (e.g. dropout layer)
	for (auto i = 0u; i < forwardOnlyLayerNum; i++)
	{
		if (i == 0)
		{
			forwardOnlyLayers[i]->forwardPropagation(input, forwardOnlyLayers[i]->getOutput());
		}
		else
		{
			forwardOnlyLayers[i]->forwardPropagation(forwardOnlyLayers[i - 1]->getOutput(), forwardOnlyLayers[i]->getOutput());
		}
	}

	// Output class if user wishes it
	if (outputEnabled && !suppressOutput)
	{
		printResults(forwardOnlyLayers.back()->getOutput());
	}

	return forwardOnlyLayers.back()->getOutput();
}


/*
 * @brief Trains the Convolutional Neural Network with given settings on given dataset
 *
 * @param  settings       Training settings (batch size, number of epochs etc.)
 * @param  trainingData   Vector of pairs (input, expected output)
 * @param  lossFunction   Loss function to be used
 * @param  optimizer      Optimizer to be used
 * @param  validationData Validation data for periodic validation
 *
 * @return lastError      Error in last epoch
 * @throws CNNException if no data were passed or if no layers were added or if training produces NaN weights
 */
float ConvolutionalNeuralNetwork::train(TrainingSettings & settings, std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & trainingData, 
	const LossFunctionType & lossFunction, const std::shared_ptr<IOptimizer> optimizer, const std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & validationData /*={}*/)
{
	if (trainingData.empty())
	{
		throw CNNException("No data to perform training on.");
	}
	else if (allLayers.empty())
	{
		throw CNNException("No layers to perform training on.");
	}
	
	suppressOutput = true;
	training = true;

	// Initialize optimizers in layers
	for (auto & layer : allLayers)
	{
		layer->setOptimizer(optimizer);
		layer->initializeOptimizer();
	}

	auto start = std::chrono::system_clock::now();

	// Validate before training if flag set
	if (settings.periodicValidation && outputEnabled)
	{
		validate(validationData);
		std::cout << std::endl;
	}

	auto trainingDataSize = trainingData.size();

	// For each epoch iterate through all the data
	auto epochError = 0.0f;
	auto batchError = 0.0f;
	for (auto epoch = 0u; epoch < settings.epochs; epoch++)
	{
		// Shuffle training dataset if chosen
		if (settings.shuffle)
		{
			std::random_shuffle(trainingData.begin(), trainingData.end());
		}
		
		epochError = 0.0f;
		auto epochStart = std::chrono::system_clock::now();

		// For each training case in each epoch compute error and update weights/filters
		for (auto s = 0u; s < trainingDataSize; s++)
		{
			// Forward propagation
			for (auto i = 0u; i < allLayerNum; i++)
			{
				if (i == 0)
				{
					allLayers[i]->forwardPropagation(trainingData[s].first, allLayers[i]->getOutput());
				}
				else
				{
					allLayers[i]->forwardPropagation(allLayers[i - 1]->getOutput(), allLayers[i]->getOutput());
				}
			}

			// Compute error
			auto errorResult = computeError(allLayers.back()->getOutput(), trainingData[s].second, lossFunction);

			epochError += errorResult.first;
			batchError += errorResult.first;
			if (isnan(errorResult.first) || isinf(errorResult.first))
			{
				throw CNNException("Output error is NaN/INF, this may be caused by invalid choice of hyperparameters.");
			}

			// Periodically output average error if set
			if (outputEnabled && settings.errorOutputRate > 0 && ((((s + 1) % settings.errorOutputRate) == 0) || (s + 1 == trainingDataSize)))
			{
				std::cout << "(" << s + 1 << "/" << trainingDataSize << "): " << batchError / settings.errorOutputRate << std::endl;
				batchError = 0.0f;
			}

			// Backward propagation (learning)
			for (auto i = static_cast<int>(allLayerNum - 1); i >= 0; i--)
			{
				if (i == static_cast<int>(allLayerNum - 1))
				{
					allLayers[i]->backwardPropagation(allLayers[i-1]->getOutput(), allLayers[i]->getOutput(), errorResult.second, allLayers[i]->getGradientOutput(), settings);
				}
				else if (i == 0)
				{
					allLayers[i]->backwardPropagation(trainingData[s].first, allLayers[i]->getOutput(), allLayers[i+1]->getGradientOutput(), allLayers[i]->getGradientOutput(), settings);
				}
				else
				{
					allLayers[i]->backwardPropagation(allLayers[i-1]->getOutput(), allLayers[i]->getOutput(), allLayers[i+1]->getGradientOutput(), allLayers[i]->getGradientOutput(), settings);
				}
			}
		}

		if (outputEnabled || onEpochFinishedCallback)
		{
			// Epoch length in seconds
			auto epochEnd = std::chrono::system_clock::now();
			std::chrono::duration<float> diff = epochEnd - epochStart;
			auto epochLength = diff.count();

			// Average error per training sample (with output if set)
			auto epochAverageError = epochError / trainingDataSize;
			if (outputEnabled && (((epoch + 1) % settings.epochOutputRate) == 0 || (epoch + 1) == settings.epochs))
			{
				std::cout << "Error in epoch " << epoch + 1 << ": " << epochAverageError << " (" << epochLength << " s)" << std::endl;
			}
			
			// Validation accuracy if set (with output if set)
			auto validationAccuracy = NAN;
			if (settings.periodicValidation)
			{
				validationAccuracy = validate(validationData);
				if (outputEnabled)
				{
					std::cout << std::endl;
				}
			}

			// If callback set, call it (logging, lerning rate scheduling etc.)
			if (onEpochFinishedCallback)
			{
				onEpochFinishedCallback(epoch + 1, settings, epochAverageError, validationAccuracy, epochLength);
			}
		}
	}

	// Output total training time if user wishes to see it
	if (outputEnabled)
	{
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<float> diff = end - start;
		std::cout << "Total training time: " << diff.count() << " s" << std::endl;
	}

	training = false;
	suppressOutput = false;

	return epochError;
}


/*
 * @brief Validates network on set of test data, returns accuracy in percents
 * 
 * @param  data        Data for validation
 *
 * @retrun accuracy    Accuracy (% for classification, average error for regression)
 * @throws CNNException if no validation data were passed
 */
float ConvolutionalNeuralNetwork::validate(const std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & data)
{
	suppressOutput = true;

	if (data.empty())
	{
		throw CNNException("No data to perform validation on.");
	}

	auto totalCnt = data.size();
	float outVal;
	switch (taskType)
	{
		// Computes average absolute and relative error per sample
		case TaskType::Regression:
		{
			auto avgAbsError = 0.0f;
			auto avgRelError = 0.0f;

			for (const auto & input : data)
			{
				auto output = run(input.first);
				auto expected = input.second;
				auto flattenedSize = output.getFlattenedSize();
				
				auto tmpAbsError = 0.0f;
				auto tmpRelError = 0.0f;
				for (auto i = 0u; i < flattenedSize; i++)
				{
					tmpAbsError += static_cast<float>(fabs(expected(i) - output(i)));
					tmpRelError += static_cast<float>(fabs(expected(i) - output(i)) / std::max(fabs(expected(i)), fabs(output(i))));
				}
				avgAbsError += tmpAbsError / flattenedSize;
				avgRelError += tmpRelError / flattenedSize;
			}

			if (outputEnabled)
			{
				std::cout << "Average absolute error per sample: " << avgAbsError / totalCnt << std::endl;
				std::cout << "Average relative difference per sample: " << avgRelError / totalCnt << " %" << std::endl;
			}

			outVal = avgRelError / totalCnt;
			break;
		}
		// Computes how many inputs were classified correctly
		case TaskType::Classification: default:
		{
			auto correctCnt = 0u;

			for (const auto & input : data)
			{
				auto output = run(input.first).getImageAsVector();
				auto expected = input.second.getImageAsVector();

				auto outClass = std::max_element(output.begin(), output.end());
				auto expClass = std::max_element(expected.begin(), expected.end());

				if ((outClass - output.begin()) == (expClass - expected.begin()))
				{
					correctCnt++;
				}
			}

			auto successRate = static_cast<float>(correctCnt) / totalCnt * 100.0f;

			if (outputEnabled)
			{
				std::cout << "Succesfully classified " << correctCnt << " out of " << totalCnt << std::endl;
				std::cout << "\tSuccess rate: " << successRate << " %" << std::endl;
				std::cout << "\tError   rate: " << 100.0f - successRate << " %" << std::endl;
			}

			outVal = successRate;
			break;
		}
	}

	suppressOutput = false;
	return outVal;
}


/*
 * @brief Print results of run
 */
void ConvolutionalNeuralNetwork::printResults(const Image<ForwardType> & output) const
{
	switch (taskType)
	{
		// Predicting values of continous functions
		case TaskType::Regression:
		{
			auto vectorOutput = output.getImageAsVector();
			for (const auto & val : vectorOutput)
			{
				std::cout << std::fixed << std::setprecision(5) << val << " ";
			}
			break;
		}
		// Predicting class of input (does not support multiclass classification)
		case TaskType::Classification: default:
		{
			auto vectorOutput = output.getImageAsVector();

			auto outClass = std::max_element(vectorOutput.begin(), vectorOutput.end());
			for (const auto & val : vectorOutput)
			{
				std::cout << std::fixed << std::setprecision(3) << val << " ";
			}

			std::cout << "\nOutput class is " << outClass - vectorOutput.begin() << "\n" << std::endl;
			break;
		}
	}
}


/*
 * @brief Computes total error and elso error vector for output layer
 */
std::pair<BackwardType, Image<BackwardType>> ConvolutionalNeuralNetwork::computeError(const Image<ForwardType> & actualOutput, 
	const Image<ForwardType> & expectedOutput, const LossFunctionType & lossFunctionType) const
{
	static auto epsilon = Limits::getEpsilonValue<BackwardType>(); // to avoid zero division or log(0)
	static auto ONE = static_cast<BackwardType>(1.0f); // shortcut

	auto flattenedSize = actualOutput.getFlattenedSize();
	Image<BackwardType> errorVector(actualOutput.getDimensions());
	auto error = static_cast<BackwardType>(0.0f);
	switch (lossFunctionType)
	{
		// For regression tasks and outputs different than <0,1>
		case LossFunctionType::MeanSquaredError: default:
		{
			for (auto i = 0u; i < flattenedSize; i++)
			{
				auto actual = static_cast<BackwardType>(actualOutput(i));
				auto expected = static_cast<BackwardType>(expectedOutput(i));

				auto diff = (actual - expected);
				errorVector(i) = static_cast<BackwardType>(2) * diff / static_cast<BackwardType>(static_cast<float>(flattenedSize));

				error += diff * diff / static_cast<BackwardType>(static_cast<float>(flattenedSize));
			}
			break;
		}
		// For classification tasks with more than two classes, requires output in range <0, 1> - use SoftMax or Sigmoid in last layer
		case LossFunctionType::CrossEntropy:
		{
			for (auto i = 0u; i < flattenedSize; i++)
			{
				auto actual = static_cast<BackwardType>(actualOutput(i));
				auto expected = static_cast<BackwardType>(expectedOutput(i));

				errorVector(i) = -expected / (actual + epsilon);

				error += -expected * static_cast<BackwardType>(log(actual + epsilon));
			}
			break;
		}
		// For classification tasks with two classes, requires output in range <0, 1> - use SoftMax or Sigmoid in last layer
		case LossFunctionType::BinaryCrossEntropy:
		{
			for (auto i = 0u; i < flattenedSize; i++)
			{
				auto actual = static_cast<BackwardType>(actualOutput(i));
				auto expected = static_cast<BackwardType>(expectedOutput(i));

				errorVector(i) = (actual - expected) / (actual * (ONE - actual) + epsilon);

				error += (-expected) * static_cast<BackwardType>(log(actual + epsilon)) - ((ONE - expected) * static_cast<BackwardType>(log(ONE - actual + epsilon)));
			}
			break;
		}
	}

	return std::make_pair(error, errorVector);
}


/*
 * @brief Set on epoch finished callback
 */
void ConvolutionalNeuralNetwork::setOnEpochFinishedCallback(OnEpochFinishedCallbackType callback)
{
	onEpochFinishedCallback = std::move(callback);
}


/*
 * @brief Enables output to std::cout
 */
void ConvolutionalNeuralNetwork::enableOutput()
{
	outputEnabled = true;
}


/*
 * @brief Disables output
 */
void ConvolutionalNeuralNetwork::disableOutput()
{
	outputEnabled = false;
}


/*
 * @brief Returns type of task this CNN is intended for
 */
TaskType ConvolutionalNeuralNetwork::getTaskType() const
{
	return taskType;
}


/*
 * @brief Returns expected input dimensions
 */
Dimensions ConvolutionalNeuralNetwork::getInputSize() const
{
	return inputSize;
}


/*
 * @brief Returns output dimensions
 */
Dimensions ConvolutionalNeuralNetwork::getOutputSize() const
{
	return outputSize;
}