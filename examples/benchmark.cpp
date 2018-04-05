/*
 * @author Petr Rek
 * @project CNN library
 * @brief Example of using the library in your own project
 */

#include "src/ConvolutionalNeuralNetwork.h"

#include "src/Parsers/IdxParser.h"

constexpr unsigned CLASSES_NUM = 10;

int main(int, char **)
{
	srand(static_cast<unsigned>(NULL));

	auto trainingData = IdxParser::parseLabelledImages("../resources/mnist/train-images.idx3-ubyte", "../resources/mnist/train-labels.idx1-ubyte", CLASSES_NUM);
	auto validationData = IdxParser::parseLabelledImages("../resources/mnist/test-images.idx3-ubyte", "../resources/mnist/test-labels.idx1-ubyte", CLASSES_NUM);

	if (trainingData.empty() || validationData.empty())
	{
		std::cerr << "Training and/or validation data could not be loaded" << std::endl;
		return EXIT_FAILURE;
	}

	auto inputDimensions = trainingData[0].first.getDimensions();

	auto layer1 = std::make_shared<Convolution>(inputDimensions, 1, 8, 5, 0, true);
	auto layer2 = std::make_shared<ReLU>(layer1->getOutputSize());
	auto layer3 = std::make_shared<MaxPooling>(layer2->getOutputSize(), 2, 2);
	auto layer4 = std::make_shared<FullyConnected>(layer3->getOutputSize(), Dimensions{ 10, 1, 1 }, true);
	auto layer5 = std::make_shared<Sigmoid>(layer4->getOutputSize());

	auto cnn = ConvolutionalNeuralNetwork(TaskType::Classification);
	cnn.addLayer(layer1);
	cnn.addLayer(layer2);
	cnn.addLayer(layer3);
	cnn.addLayer(layer4);
	cnn.addLayer(layer5);

	TrainingSettings settings;
	settings.epochs = 10;
	settings.batchSize = 1;
	settings.epochOutputRate = 1;
	settings.errorOutputRate = 0;
	settings.periodicValidation = false;
	settings.shuffle = false;

	cnn.enableOutput();

	auto optimizer = std::make_shared<SgdWithMomentum>();
	optimizer->learningRate = 0.01f;
	optimizer->momentum = 0.9f;
	optimizer->weightDecay = 0.0f;

	cnn.train(settings, trainingData, LossFunctionType::MeanSquaredError, optimizer);

	cnn.validate(validationData);

	return EXIT_SUCCESS;
}