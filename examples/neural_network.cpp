/*
 * @author Petr Rek
 * @project CNN library
 * @brief Example of using the library in your own project
 */

#include "src/NeuralNetwork.h"

#include "src/Parsers/IdxParser.h"

#include <ctime>
#include <string>

constexpr unsigned CLASSES_NUM = 10;

int main(int, char **)
{
	srand(static_cast<unsigned>(time(NULL)));

	// Load training and validation data
	auto trainingData = IdxParser::parseLabelledImages("../resources/mnist/train-images.idx3-ubyte", "../resources/mnist/train-labels.idx1-ubyte", CLASSES_NUM);
	auto validationData = IdxParser::parseLabelledImages("../resources/mnist/test-images.idx3-ubyte", "../resources/mnist/test-labels.idx1-ubyte", CLASSES_NUM);

	if (trainingData.empty() || validationData.empty())
	{
		std::cerr << "Training and/or validation data could not be loaded" << std::endl;
		return EXIT_FAILURE;
	}

	// Flatten 3D image to 1D array (since we do not have any 1D data)
	std::vector<std::pair<std::vector<float>, std::vector<float>>> flattenedTrainingData;
	for (const auto & img : trainingData)
	{
		flattenedTrainingData.emplace_back(img.first.getImageAsVector(), img.second.getImageAsVector());
	}
	std::vector<std::pair<std::vector<float>, std::vector<float>>> flattenedValidationData;
	for (const auto & img : validationData)
	{
		flattenedValidationData.emplace_back(img.first.getImageAsVector(), img.second.getImageAsVector());
	}

	// Create neural network
	auto inputSize = trainingData[0].first.getFlattenedSize();
	auto nn = NeuralNetwork(
		inputSize, // input size
		{{128, ActivationFunction::Tanh}, {64, ActivationFunction::Tanh}}, // hidden layers
		{10, ActivationFunction::SoftMax}, // output layer
		true, // use bias 
		TaskType::Classification // task type
	);

	// Training settings
	TrainingSettings settings;
	settings.epochs = 5;
	settings.batchSize = 1;
	settings.epochOutputRate = 1;
	settings.errorOutputRate = 10000;
	settings.periodicValidation = true;
	settings.shuffle = true;

	auto optimizer = std::make_shared<SgdWithMomentum>();
	optimizer->learningRate = 0.001f;
	optimizer->momentum = 0.8f;
	optimizer->weightDecay = 0.001f;

	// Run training and validation
	nn.enableOutput();

	nn.train(settings, flattenedTrainingData, LossFunctionType::CrossEntropy, optimizer, flattenedValidationData);

	return EXIT_SUCCESS;
}