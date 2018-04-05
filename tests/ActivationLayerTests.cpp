/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Unit tests for Activation layer
 */

#include <gtest/gtest.h>

#include "src/Image.h"
#include "src/Layers/ReluActivationLayer.h"

TEST(ActivationLayerTest, ReluWorksCorrectlyOn2DImage)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 1, 2, 3, -4 },
		{ 5, -6, 7, 8 },
		{ 9, -10, 11, 12 },
		{ 13, 14, -15, -16 }
	} };

	auto img = Image<ForwardType>(input);

	ReluActivationLayer<ForwardType, WeightType> activationLayer(img.getDimensions());

	activationLayer.forwardPropagation(img, activationLayer.getOutput());

	std::vector<std::vector<std::vector<ForwardType>>> expectedOutput =
	{ { { 1, 2, 3, 0 },
		{ 5, 0, 7, 8 },
		{ 9, 0, 11, 12 },
		{ 13, 14, 0, 0 }
	} };

	EXPECT_TRUE(Image<ForwardType>(expectedOutput) == activationLayer.getOutput());
}

TEST(ActivationLayerTest, ReluWorksCorrectlyOn3DImage)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 1, -5 },
		{ -3, 4 } },
		{ { 5, -6 },
		{ 13, -8 }
	} };

	auto img = Image<ForwardType>(input);

	ReluActivationLayer<ForwardType, WeightType> activationLayer(img.getDimensions());

	activationLayer.forwardPropagation(img, activationLayer.getOutput());

	std::vector<std::vector<std::vector<ForwardType>>> expectedOutput =
	{ { { 1, 0 },
		{ 0, 4 } },
		{ { 5, 0 },
		{ 13, 0 }
	} };

	EXPECT_TRUE(Image<ForwardType>(expectedOutput) == activationLayer.getOutput());
}
