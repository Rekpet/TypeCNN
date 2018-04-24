/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Unit tests for Fully connected layer
 */

#include <gtest/gtest.h>

#include "src/Image.h"
#include "src/Layers/FullyConnectedLayer.h"

// We need to access inner structures of Convolutional layer for some tests
class FullyConnectedLayerTests : public ::testing::Test, public FullyConnectedLayer<ForwardType, WeightType>
{
	public:
	
		FullyConnectedLayerTests()
		: FullyConnectedLayer<ForwardType, WeightType>(Dimensions{ 3, 1, 1 }, Dimensions{ 2, 1, 1 }, true)
		{
		}
	
		virtual void SetUp() override 
		{
		};

		virtual void TearDown() override
		{
		};
};

TEST_F(FullyConnectedLayerTests, ForwardPropagationWithBias)
{
	auto layer = std::make_shared<FullyConnectedLayer<ForwardType, WeightType>>(Dimensions{ 3, 1, 1 }, Dimensions{ 2, 1, 1 }, true);

	Image<ForwardType> input(std::vector<std::vector<std::vector<ForwardType>>>{ {
		{ 1.0f, 2.0f, 3.0f }
		} });

	Image<WeightType> testWeights({ {
		{ -3.0f, -2.0f, -1.0f, 0.0f },
		{  1.0f,  2.0f,  3.0f, 4.0f }
		} });

	Image<ForwardType> expectedOutput(std::vector<std::vector<std::vector<ForwardType>>>{ {
		{ -10.0f, 18.0f }
		} }
	);

	layer->setNeuronWeights(testWeights);
	layer->forwardPropagation(input, layer->getOutput());

	EXPECT_TRUE(expectedOutput == layer->getOutput());
}

TEST_F(FullyConnectedLayerTests, ForwardPropagationWithoutBias)
{
	auto layer = std::make_shared<FullyConnectedLayer<ForwardType, WeightType>>(Dimensions{ 3, 1, 1 }, Dimensions{ 2, 1, 1 }, false);

	Image<ForwardType> input(std::vector<std::vector<std::vector<ForwardType>>>{ {
		{ 1.0f, 2.0f, 3.0f }
		} });

	Image<WeightType> testWeights({ {
		{ -3.0f, -2.0f, -1.0f, 10.0f },
		{  1.0f,  2.0f,  3.0f, 10.0f }
		} });

	Image<ForwardType> expectedOutput(std::vector<std::vector<std::vector<ForwardType>>>{ {
		{ -10.0f, 14.0f }
		} }
	);

	layer->setNeuronWeights(testWeights);
	layer->forwardPropagation(input, layer->getOutput());

	EXPECT_TRUE(expectedOutput == layer->getOutput());
}

TEST_F(FullyConnectedLayerTests, BackwardPropagation)
{
	// Input values
	
	Image<ForwardType> input(std::vector<std::vector<std::vector<ForwardType>>>{ {
		{ 1.0f, 2.0f, 3.0f }
	} });

	Image<WeightType> testWeights({ {
		{  1.0f,  2.0f,  3.0f, 10.0f },
		{ -3.0f, -2.0f, -1.0f, -10.0f }
		} });

	Image<BackwardType> inputDeltas(std::vector<std::vector<std::vector<BackwardType>>>{{
		{-1.0f, 2.0f}
	}});

	// Expected values

	Image<BackwardType> expectedDeltas({ {
		{  -1.0f,  -2.0f,  -3.0f, -1.0f },
		{ 2.0f, 4.0f, 6.0f, 2.0f }
		} });

	Image<BackwardType> expectedOutputDeltas(std::vector<std::vector<std::vector<BackwardType>>>{{
		{ -7.0f, -6.0f, -5.0f},
	}});

	// Run and test

	setNeuronWeights(testWeights);

	TrainingSettings settings;
	settings.batchSize = 10; // to not update weights
	backwardPropagation(input, getOutput(), inputDeltas, getGradientOutput(), settings);

	EXPECT_TRUE(expectedOutputDeltas == getGradientOutput());
	EXPECT_TRUE(expectedDeltas == deltas);
}
