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
		: FullyConnectedLayer<ForwardType, WeightType>(Dimensions{10, 1, 1}, Dimensions{ 10, 1, 1 }, true)
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
