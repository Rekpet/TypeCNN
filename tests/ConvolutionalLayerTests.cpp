/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Unit tests for Convolutional layer
 */

#include <gtest/gtest.h>

#include "src/Image.h"
#include "src/Layers/ConvolutionalLayer.h"

// We need to access inner structures of Convolutional layer for some tests
class ConvolutionalLayerTests : public ::testing::Test, public ConvolutionalLayer<ForwardType, WeightType>
{
	public:
	
		ConvolutionalLayerTests() 
		: ConvolutionalLayer<ForwardType, WeightType>(Dimensions{5, 5, 1}, 2, 1, 3, 0, true)
		{
		}
	
		virtual void SetUp() override 
		{
		};

		virtual void TearDown() override
		{
		};
};

TEST_F(ConvolutionalLayerTests, ConvolutionWorksCorrectlyOn3DImageWithZeroPadding)
{
	Image<ForwardType> input(std::vector<std::vector<std::vector<ForwardType>>>
	{ { { 1 } },
	  { { 5 }
	} });

	Image<BackwardType> filter1(
	{ { { 1, 2, 3 },
		{ 4, 5, 6 },
		{ 7, 8, 9 } },
	  { { 1, 2, 3 },
		{ 4, 5, 6 },
		{ 7, 8, 9 }
	} });

	Image<BackwardType> filter2(
	{ { { 1, 1, 1 },
	    { 1, 1, 1 },
	    { 1, 1, 1 } },
	  { { 1, 1, 1 },
	    { 1, 1, 1 },
	    { 1, 1, 1 },
	} });

	Image<BackwardType> filter3(
	{ { { 2, 2, 2 },
	    { 2, 2, 2 },
	    { 2, 2, 2 } },
	  { { 2, 2, 2 },
	    { 2, 2, 2 },
	    { 2, 2, 2 }
	} });

	auto testFilters = { filter1, filter2, filter3 };

	ConvolutionalLayer<ForwardType, WeightType> convolutionaLayer(input.getDimensions(), 1, 3, 3, 1, false);
	convolutionaLayer.loadFilters(testFilters);

	convolutionaLayer.forwardPropagation(input, convolutionaLayer.getOutput());

	Image<ForwardType> expectedOutput(std::vector<std::vector<std::vector<ForwardType>>>
	{ { { 30 } },
	  { { 6 } },
	  { { 12 } }
	});

	EXPECT_TRUE(expectedOutput == convolutionaLayer.getOutput());
}

TEST_F(ConvolutionalLayerTests, ConvolutionWorksCorrectlyOn3DImageWithZeroPaddingAndBias)
{
	Image<ForwardType> input(std::vector<std::vector<std::vector<ForwardType>>>
	{ { { 1 } },
	  { { 5 }
	} });

	Image<BackwardType> filter1(
	{ { { 1, 2, 3 },
		{ 4, 5, 6 },
		{ 7, 8, 9}},
	  { { 1, 2, 3 },
		{ 4, 5, 6 },
		{ 7, 8, 9 }
	} });

	Image<BackwardType> filter2(
	{ { { 1, 1, 1 },
		{ 1, 1, 1 },
		{ 1, 1, 1 } },
	  { { 1, 1, 1 },
		{ 1, 1, 1 },
		{ 1, 1, 1 },
	} });

	Image<BackwardType> filter3(
	{ { { 2, 2, 2 },
	    { 2, 2, 2 },
	    { 2, 2, 2 } },
	  { { 2, 2, 2 },
	    { 2, 2, 2 },
	    { 2, 2, 2 }
	} });

	auto testFilters = { filter1, filter2, filter3 };

	ConvolutionalLayer<ForwardType, WeightType> convolutionaLayer(input.getDimensions(), 1, 3, 3, 1, true);
	convolutionaLayer.loadFilters(testFilters, {1, 2, 3});

	convolutionaLayer.forwardPropagation(input, convolutionaLayer.getOutput());

	Image<ForwardType> expectedOutput(std::vector<std::vector<std::vector<ForwardType>>>
	{ { { 31 } },
	  { { 8 } },
	  { { 15 } }
	});

	EXPECT_TRUE(expectedOutput == convolutionaLayer.getOutput());
}

TEST_F(ConvolutionalLayerTests, ForwardAndBackwardPropagationWorkCorrectly)
{
	// https://github.com/tiny-dnn/tiny-dnn

	// Input image
	Image<ForwardType> input({{
		{0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, 
		{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 
		{2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
		{3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, 
		{4.0f, 5.0f, 6.0f, 7.0f, 8.0f}
	}});

	// Initial filter values and bias values
	Image<WeightType> filter({{
		{0.5f, 0.5f, 0.5f}, 
		{0.5f, 0.5f, 0.5f}, 
		{0.5f, 0.5f, 0.5f}
	}});
	std::vector<Image<WeightType>> testFilters = { filter };
	std::vector<WeightType> testBiases = { 0.5f };

	// Expected output of forward propagation
	Image<ForwardType> expectedOutput({{
		{9.5f, 18.5f},
		{18.5f, 27.5f}
	}});

	// Input deltas to convolutional layer
	Image<BackwardType> inputDeltas({{
		{-1.0f, 2.0f},
		{3.0f, 0.0f}
	}});

	// Expected output deltas for previous layer
	Image<BackwardType> expectedOutputDeltas({{
		{-0.5f, -0.5f, 0.5f, 1.0f, 1.0f}, 
		{-0.5f, -0.5f, 0.5f, 1.0f, 1.0f}, 
		{1.0f, 1.0f, 2.0f, 1.0f, 1.0f},
		{1.5f, 1.5f, 1.5f, 0.0f, 0.0f}, 
		{1.5f, 1.5f, 1.5f, 0.0f, 0.0f}
	}});

	// Expected deltas for filter
	Image<BackwardType> expectedFilterDeltas({{
		{10.0f, 14.0f, 18.0f},
		{14.0f, 18.0f, 22.0f},
		{18.0f, 22.0f, 26.0f}
	}});

	// Expected bias deltas
	std::vector<BackwardType> expectedBiasDeltas = { 4.0f };

	// Set up layer
	loadFilters(testFilters, testBiases);

	// Forward propagate and check output
	forwardPropagation(input, getOutput());

	EXPECT_TRUE(expectedOutput == getOutput());

	// Backward propagate and check gradients
	TrainingSettings settings;
	settings.batchSize = 10; // to not update weights
	backwardPropagation(input, getOutput(), inputDeltas, getGradientOutput(), settings);

	EXPECT_TRUE(expectedOutputDeltas == getGradientOutput());
	EXPECT_TRUE(expectedFilterDeltas == filterDeltas[0]);
	EXPECT_EQ(expectedBiasDeltas[0], biasDeltas[0]);
}
