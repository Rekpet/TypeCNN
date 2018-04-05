/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Unit tests for Pooling layer
 */

#include <gtest/gtest.h>

#include "src/Image.h"
#include "src/Layers/MaxPoolingLayer.h"
#include "src/Layers/AvgPoolingLayer.h"

TEST(PoolingLayerTest, MaxWorksCorrectlyOnSimpleImage)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 16 }
	} };

	auto img = Image<ForwardType>(input);
	MaxPoolingLayer<ForwardType, WeightType> poolingLayer(img.getDimensions(), 2, 2);
	poolingLayer.forwardPropagation(img, poolingLayer.getOutput());

	std::vector<std::vector<std::vector<ForwardType>>> expectedOutput =
	{ { {6, 8},
		{14, 16}
	} };

	EXPECT_TRUE(Image<ForwardType>(expectedOutput) == poolingLayer.getOutput());
}


TEST(PoolingLayerTest, MaxWorksCorrectlyOn3DImage)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 1, 5 },
		{ 3, 4 }
		},
		{ { 5, 6 },
		{ 13, 8 }
	} };

	auto img = Image<ForwardType>(input);
	MaxPoolingLayer<ForwardType, WeightType> poolingLayer(img.getDimensions(), 2, 2);
	poolingLayer.forwardPropagation(img, poolingLayer.getOutput());

	std::vector<std::vector<std::vector<ForwardType>>> expectedOutput =
	{ { { 5 }
	  },
	  { { 13}
	} };

	EXPECT_TRUE(Image<ForwardType>(expectedOutput) == poolingLayer.getOutput());
}

TEST(PoolingLayerTest, BackpropagationOnMaxWorksCorrectlyOn3DImage)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 1, 5 },
		{ 3, 4 }
		},
	  { { 5, 6 },
		{ 13, 8 }
	  } };

	std::vector<std::vector<std::vector<BackwardType>>> error =
	{ { { 8 }
		},
	  { { 8 }
		} };

	std::vector<std::vector<std::vector<BackwardType>>> expectedError =
	{ { { 0, 8 },
	    { 0, 0 }
		},
	  { { 0, 0 },
		{ 8, 0 }
	  } };

	auto in = Image<ForwardType>(input);
	auto err = Image<ForwardType>(error);

	MaxPoolingLayer<ForwardType, WeightType> poolingLayer(in.getDimensions(), 2, 2);

	poolingLayer.forwardPropagation(in, poolingLayer.getOutput());

	poolingLayer.backwardPropagation(in, poolingLayer.getOutput(), err, poolingLayer.getGradientOutput(), TrainingSettings{});

	EXPECT_TRUE(Image<BackwardType>(expectedError) == poolingLayer.getGradientOutput());
}


TEST(PoolingLayerTest, AvgWorksCorrectlyOnSimpleImage)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 1, 1, 2, 2 },
		{ 1, 1, 2, 2 },
		{ 3, 3, 10, 10 },
		{ 3, 3, 14, 14 }
	} };

	auto img = Image<ForwardType>(input);

	AvgPoolingLayer<ForwardType, WeightType> poolingLayer(img.getDimensions(), 2, 2);
	poolingLayer.forwardPropagation(img, poolingLayer.getOutput());

	std::vector<std::vector<std::vector<ForwardType>>> expectedOutput =
	{ { { 1, 2 },
		{ 3, 12 }
	} };

	EXPECT_TRUE(Image<ForwardType>(expectedOutput) == poolingLayer.getOutput());
}

TEST(PoolingLayerTest, ThrowsWhenStrideIsZero)
{
	EXPECT_THROW((MaxPoolingLayer<ForwardType, WeightType>({6, 6, 6}, 6, 0)), PoolingLayerException);
}

TEST(PoolingLayerTest, ThrowsWhenExtentIsLargerThanImage)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 6, 8 },
		{ 14, 16 }
	} };

	auto img = Image<ForwardType>(input);

	EXPECT_THROW((MaxPoolingLayer<ForwardType, WeightType>(img.getDimensions(), 3, 6)), PoolingLayerException);
}

TEST(PoolingLayerTest, ThrowsWhenExtentCannotBeApplied)
{
	std::vector<std::vector<std::vector<ForwardType>>> input =
	{ { { 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 16 }
	} };

	auto img = Image<ForwardType>(input);

	EXPECT_THROW((AvgPoolingLayer<ForwardType, WeightType>(img.getDimensions(), 3, 6)), PoolingLayerException);
}
