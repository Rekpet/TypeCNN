/*
 * @author Petr Rek
 * @project CNN library
 * @brief Example of using different types across different layers
 */

#include "src/Layers/ConvolutionalLayer.h"
#include "src/Layers/FullyConnectedLayer.h"
#include "src/Layers/ActivationLayer.h"
#include "src/Layers/PoolingLayer.h"
#include "src/Layers/ConversionLayer.h"

#include "src/ConvolutionalNeuralNetwork.h"

#include "src/Parsers/IdxParser.h"

#include "src/Utils/FixedPointNumber.h"

constexpr unsigned CLASSES_NUM = 10;

using InputType = FixedPoint<14, 14>;
using MiddleType = FixedPoint<6, 10>;
using OutputType = FixedPoint<8, 8>;

class FixedPointCNN
{

public:

	FixedPointCNN(const Dimensions & inputSize)
	{
		layer1 = std::make_shared<ConvolutionalLayer<InputType, InputType>>(inputSize, 1, 8, 5, 0, true);
		layer2 = std::make_shared<ReluActivationLayer<InputType, InputType>>(layer1->getOutputSize());
		layer3 = std::make_shared<ConversionLayer<InputType, MiddleType>>(layer2->getOutputSize());
		layer4 = std::make_shared<ConvolutionalLayer<MiddleType, MiddleType>>(layer3->getOutputSize(), 1, 8, 5, 0, true);
		layer5 = std::make_shared<ReluActivationLayer<MiddleType, MiddleType>>(layer4->getOutputSize());
		layer6 = std::make_shared<MaxPoolingLayer<MiddleType, MiddleType>>(layer5->getOutputSize(), 2, 2);
		layer7 = std::make_shared<ConversionLayer<MiddleType, OutputType>>(layer6->getOutputSize());
		layer8 = std::make_shared<FullyConnectedLayer<OutputType, OutputType>>(layer7->getOutputSize(), Dimensions{ 10, 1, 1 }, true);
		layer9 = std::make_shared<SigmoidActivationLayer<OutputType, OutputType>>(layer8->getOutputSize());
	}

	Image<OutputType> run(const Image<InputType> & input)
	{
		layer1->forwardPropagation(input, layer1->getOutput());
		layer2->forwardPropagation(layer1->getOutput(), layer2->getOutput());
		layer3->forwardPropagation(layer2->getOutput(), layer3->getOutput());
		layer4->forwardPropagation(layer3->getOutput(), layer4->getOutput());
		layer5->forwardPropagation(layer4->getOutput(), layer5->getOutput());
		layer6->forwardPropagation(layer5->getOutput(), layer6->getOutput());
		layer7->forwardPropagation(layer6->getOutput(), layer7->getOutput());
		layer8->forwardPropagation(layer7->getOutput(), layer8->getOutput());
		layer9->forwardPropagation(layer8->getOutput(), layer9->getOutput());

		return layer9->getOutput();
	}

	void train(const TrainingSettings & settings, const std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & data, const std::shared_ptr<IOptimizer> optimizer)
	{
		if (data.empty())
		{
			std::cerr << "No data loaded from training data set." << std::endl;
			return;
		}

		layer1->setOptimizer(optimizer); layer1->initializeOptimizer();
		layer2->setOptimizer(optimizer); layer2->initializeOptimizer();
		layer4->setOptimizer(optimizer); layer4->initializeOptimizer();
		layer5->setOptimizer(optimizer); layer5->initializeOptimizer();
		layer6->setOptimizer(optimizer); layer6->initializeOptimizer();
		layer8->setOptimizer(optimizer); layer8->initializeOptimizer();
		layer9->setOptimizer(optimizer); layer9->initializeOptimizer();

		auto inputImage = Image<InputType>{ data[0].first.getDimensions() };
		auto expectedOutput = Image<OutputType>{ data[0].second.getDimensions() };

		for (auto epoch = 0u; epoch < settings.epochs; epoch++)
		{
			auto epochError = 0.0f;
			// For each training case in each epoch compute error and update weights/filters
			for (const auto & input : data)
			{
				convertImage<ForwardType, InputType>(input.first, inputImage);
				convertImage<ForwardType, OutputType>(input.second, expectedOutput);

				auto output = run(inputImage);

				auto errorResult = computeError(output, expectedOutput);
				epochError += errorResult.first;

				if (isnan(errorResult.first) || isinf(errorResult.first))
				{
					std::cerr << "Output error is NaN/INF, this may be caused by invalid choice of hyperparameters." << std::endl;
					return;
				}

				layer9->backwardPropagation(layer8->getOutput(), layer9->getOutput(), errorResult.second, layer9->getGradientOutput(), settings);
				layer8->backwardPropagation(layer7->getOutput(), layer8->getOutput(), layer9->getGradientOutput(), layer8->getGradientOutput(), settings);
				layer7->backwardPropagation(layer6->getOutput(), layer7->getOutput(), layer8->getGradientOutput(), layer7->getGradientOutput(), settings);
				layer6->backwardPropagation(layer5->getOutput(), layer6->getOutput(), layer7->getGradientOutput(), layer6->getGradientOutput(), settings);
				layer5->backwardPropagation(layer4->getOutput(), layer5->getOutput(), layer6->getGradientOutput(), layer5->getGradientOutput(), settings);
				layer4->backwardPropagation(layer3->getOutput(), layer4->getOutput(), layer5->getGradientOutput(), layer4->getGradientOutput(), settings);
				layer3->backwardPropagation(layer2->getOutput(), layer3->getOutput(), layer4->getGradientOutput(), layer3->getGradientOutput(), settings);
				layer2->backwardPropagation(layer1->getOutput(), layer2->getOutput(), layer3->getGradientOutput(), layer2->getGradientOutput(), settings);
				layer1->backwardPropagation(inputImage, layer1->getOutput(), layer2->getGradientOutput(), layer1->getGradientOutput(), settings);
			}

			// Output current error if user wishes to see it
			if ((epoch % settings.epochOutputRate) == 0 || (epoch + 1) == settings.epochs)
			{
				std::cout << "Error in epoch: " << epoch + 1 << " is: " << epochError / data.size() << std::endl;
			}

		}
	}

	void validate(const std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> & data)
	{
		if (data.empty())
		{
			std::cerr << "No data loaded from validation data set." << std::endl;
			return;
		}

		auto totalCnt = data.size();

		auto inputImage = Image<InputType>{ data[0].first.getDimensions() };
		auto expectedOutput = Image<OutputType>{ data[0].second.getDimensions() };

		auto correctCnt = 0u;

		for (const auto & input : data)
		{
			convertImage<ForwardType, InputType>(input.first, inputImage);
			convertImage<ForwardType, OutputType>(input.second, expectedOutput);

			auto output = run(inputImage).getImageAsVector();
			auto expected = expectedOutput.getImageAsVector();

			auto outClass = std::max_element(output.begin(), output.end());
			auto expClass = std::max_element(expected.begin(), expected.end());

			if ((outClass - output.begin()) == (expClass - expected.begin()))
			{
				correctCnt++;
			}
		}

		auto successRate = static_cast<float>(correctCnt) / totalCnt * 100.0f;

		std::cout << "Succesfully classified " << correctCnt << " out of " << totalCnt << std::endl;
		std::cout << "\tSuccess rate: " << successRate << " %" << std::endl;
		std::cout << "\tError   rate: " << 100.0f - successRate << " %" << std::endl;
	}

private:

	template <class OldType, class NewType>
	void convertImage(const Image<OldType> & oldImage, Image<NewType> & newImage)
	{
		auto flattenedSize = oldImage.getFlattenedSize();

		for (auto i = 0u; i < flattenedSize; i++)
		{
			newImage(i) = static_cast<NewType>(static_cast<float>(oldImage(i)));
		}
	}

	std::pair<float, Image<BackwardType>> computeError(const Image<OutputType> & output, const Image<OutputType> & expectedOutput)
	{
		auto error = 0.0f;

		auto flattenedSize = output.getFlattenedSize();

		Image<BackwardType> errorVector(output.getDimensions());

		for (auto i = 0u; i < flattenedSize; i++)
		{
			auto diff = static_cast<BackwardType>(output(i) - expectedOutput(i));
			errorVector(i, 0, 0) = 2 * diff / flattenedSize;
			error += diff * diff / flattenedSize;
		}

		return std::make_pair(error, errorVector);
	}

private:

	std::shared_ptr<ConvolutionalLayer<InputType, InputType>> layer1;
	std::shared_ptr<ActivationLayer<InputType, InputType>> layer2;
	std::shared_ptr<ConversionLayer<InputType, MiddleType>> layer3;
	std::shared_ptr<ConvolutionalLayer<MiddleType, MiddleType>> layer4;
	std::shared_ptr<ActivationLayer<MiddleType, MiddleType>> layer5;
	std::shared_ptr<PoolingLayer<MiddleType, MiddleType>> layer6;
	std::shared_ptr<ConversionLayer<MiddleType, OutputType>> layer7;
	std::shared_ptr<FullyConnectedLayer<OutputType, OutputType>> layer8;
	std::shared_ptr<ActivationLayer<OutputType, OutputType>> layer9;

};

int main(int, char **)
{
	auto trainingData = IdxParser::parseLabelledImages("../Thesis/resources/mnist/train-images.idx3-ubyte", "../Thesis/resources/mnist/train-labels.idx1-ubyte", CLASSES_NUM, 0, 100);
	auto validationData = IdxParser::parseLabelledImages("../Thesis/resources/mnist/test-images.idx3-ubyte", "../Thesis/resources/mnist/test-labels.idx1-ubyte", CLASSES_NUM, 0, 100);

	if (trainingData.empty() || validationData.empty())
	{
		std::cerr << "Training and/or validation data could not be loaded" << std::endl;
		return EXIT_FAILURE;
	}

	auto inputDimensions = trainingData[0].first.getDimensions();

	auto cnn = FixedPointCNN(inputDimensions);

	TrainingSettings settings;
	settings.epochs = 20;
	settings.batchSize = 1;
	settings.epochOutputRate = 1;
	settings.errorOutputRate = 10000;
	settings.periodicValidation = true;

	auto optimizer = std::make_shared<SgdWithMomentum>();
	optimizer->learningRate = 0.01f;
	optimizer->momentum = 0.6f;
	optimizer->weightDecay = 0.001f;

	cnn.train(settings, trainingData, optimizer);

	cnn.validate(validationData);

	return 0;
}