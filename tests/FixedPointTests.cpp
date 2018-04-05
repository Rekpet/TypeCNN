/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Unit tests for Fixed point representation
 */

#include <gtest/gtest.h>

#include "src/Utils/FixedPointNumber.h"

#include "src/Layers/ConversionLayer.h"

TEST(FixedPointTest, CheckThatLowestPossibleRepresentationIsCorrect)
{
	/*
	 * Able to show -1.0, -0.5, 0, 0.5
	 */
	auto TwoTwo = FixedPoint<1, 1>(10.0f);
	EXPECT_EQ(TwoTwo.toFloat(), 0.5f);
	TwoTwo = -10.0f;
	EXPECT_EQ(TwoTwo.toFloat(), -1.0f);

	/*
	 * Able to show -2, -1, 0, 1
	 */
	auto TwoZero = FixedPoint<2, 0>(10.0f);
	EXPECT_EQ(TwoZero.toFloat(), 1.0f);
	TwoZero = -10.0f;
	EXPECT_EQ(TwoZero.toFloat(), -2.0f);
}


TEST(FixedPointTest, FloatAreConvertedCorrectlyToFixedPointRepresentation)
{
	/*
	 * Able to show -2 to 1.75
	 */
	auto TwoTwo = FixedPoint<2, 2>(0.0f);
	EXPECT_EQ(TwoTwo.toFloat(), 0.0f);

	TwoTwo = 0.75f;
	EXPECT_EQ(TwoTwo.toFloat(), 0.75f);

	TwoTwo = 0.875f;
	EXPECT_EQ(TwoTwo.toFloat(), 0.75f);

	TwoTwo = 4.0f;
	EXPECT_EQ(TwoTwo.toFloat(), 1.75f);

	TwoTwo = -4.0f;
	EXPECT_EQ(TwoTwo.toFloat(), -2.0f);
	
	/*
	 * Able to show -8 to 7
	 */
	auto FourZero = FixedPoint<4, 0>(0.0f);
	EXPECT_EQ(FourZero.toFloat(), 0.0f);

	FourZero = 4.999999f;
	EXPECT_EQ(FourZero.toFloat(), 4.0f);

	FourZero = -4.999999f;
	EXPECT_EQ(FourZero.toFloat(), -4.0f);

	FourZero = 16.54654f;
	EXPECT_EQ(FourZero.toFloat(), 7.0f);

	FourZero = -1548.4848f;
	EXPECT_EQ(FourZero.toFloat(), -8.0f);

	FourZero = -20.0f;
	EXPECT_EQ(FourZero.toFloat(), -8.0f);

	/*
	 * Able to show -1 to 0.875
	 */
	auto OneThree = FixedPoint<1, 3>(0.0f);
	EXPECT_EQ(OneThree.toFloat(), 0.0f);

	OneThree = 1.0f;
	EXPECT_EQ(OneThree.toFloat(), 0.875f);

	OneThree = 0.3769895f;
	EXPECT_EQ(OneThree.toFloat(), 0.375f);

	OneThree = 0.25989f;
	EXPECT_EQ(OneThree.toFloat(), 0.25f);

	OneThree = -0.875f;
	EXPECT_EQ(OneThree.toFloat(), -0.875f);

	OneThree = -1.5f;
	EXPECT_EQ(OneThree.toFloat(), -1.0f);

	/*
	 * Able to show -8 to 7.9375
	 */
 	auto FourFour = FixedPoint<4, 4>(0.0f);
	EXPECT_EQ(FourFour.toFloat(), 0.0f);

	FourFour = 16.0f;
	EXPECT_EQ(FourFour.toFloat(), 7.9375f);

	FourFour = -16.0f;
	EXPECT_EQ(FourFour.toFloat(), -8.0f);

	FourFour = 7.9375f;
	EXPECT_EQ(FourFour.toFloat(), 7.9375f);

	FourFour = -8.9375f;
	EXPECT_EQ(FourFour.toFloat(), -8.0f);

	/*
	 * Able to show -32768 to 32767.9999847412109375
	 * Do not go over 14, 14
	 */
	auto SixteenSixteen = FixedPoint<16, 16>(0.0f);
	EXPECT_EQ(SixteenSixteen.toFloat(), 0.0f);

	SixteenSixteen = 100.9999847412109375f;
	EXPECT_EQ(SixteenSixteen.toFloat(), 100.9999847412109375f);

	SixteenSixteen = -100.9999847412109375f;
	EXPECT_EQ(SixteenSixteen.toFloat(), -100.9999847412109375f);
}

TEST(ConversionLayerTest, ConversionLayerWorksProperly)
{
	// Able to show <-8, 7.9375>
	auto in = Image<FixedPoint<4, 4>>(Dimensions{ 2, 2, 1 });
	// Able to show <-2, 1.75>
	auto out = Image<FixedPoint<2, 2>>(Dimensions{ 2, 2, 1 });

	in(0, 0) = 0.0f;
	in(0, 1) = 1.5f;
	in(1, 0) = -16.0f;
	in(1, 1) = 1.375;

	EXPECT_FLOAT_EQ(in(0, 0), 0.0f);
	EXPECT_FLOAT_EQ(in(0, 1), 1.5f);
	EXPECT_FLOAT_EQ(in(1, 0), -8.0f);
	EXPECT_FLOAT_EQ(in(1, 1), 1.375f);

	auto conversionLayer = ConversionLayer<FixedPoint<4, 4>, FixedPoint<2, 2>>(in.getDimensions());
	conversionLayer.forwardPropagation(in, out);

	EXPECT_FLOAT_EQ(out(0, 0), 0.0f);
	EXPECT_FLOAT_EQ(out(0, 1), 1.5f);
	EXPECT_FLOAT_EQ(out(1, 0), -2.0f);
	EXPECT_FLOAT_EQ(out(1, 1), 1.25f);
}

TEST(FixedPointTest, FixedPointOperationsWorkCorrectly)
{
	// Various arithmetic operations that are used in implementation are tested here to be sure they work correctly

	FixedPoint<8,4> x(2);
	EXPECT_EQ(x.toFloat(), 2.0);

	FixedPoint<8,4> z(x + x);
	EXPECT_EQ(z.toFloat(), 4.0);

	z = x / x;
	EXPECT_EQ(z.toFloat(), 1.0);

	z = x * x;
	EXPECT_EQ(z.toFloat(), 4.0);

	z = x * FixedPoint<8, 4>(4);
	EXPECT_EQ(z.toFloat(), 8.0);

	z = FixedPoint<8, 4>(2) / FixedPoint<8, 4>(4);
	EXPECT_EQ(z.toFloat(), 0.5);

	z = 0.5;
	EXPECT_EQ(z.toFloat(), 0.5);

	z = z * z;
	EXPECT_EQ(z.toFloat(), 0.25);

	z = FixedPoint<8, 4>(1) / FixedPoint<8, 4>(8);
	EXPECT_EQ(z.toFloat(), 0.125);

	z = z * FixedPoint<8, 4>(100);
	EXPECT_EQ(z.toFloat(), 12.5);

	z /= z;
	EXPECT_EQ(z.toFloat(), 1.0);

	FixedPoint<2, 2> y(1.51f);
	EXPECT_EQ(y.toFloat(), 1.5);

	FixedPoint<2, 3> f(1.75f);
	EXPECT_EQ(f.toFloat(), 1.75);

	f = 1.89f;
	EXPECT_EQ(f.toFloat(), 1.875);

	f = -f;
	EXPECT_EQ(f.toFloat(), -1.875);

	f += f;
	EXPECT_EQ(f.toFloat(), -2.0f);

	f -= f / FixedPoint<8, 4>(2);
	EXPECT_EQ(f.toFloat(), -1.0f);

	FixedPoint<3, 8> a(-4.0f);
	EXPECT_EQ(a.toFloat(), -4.0f);
	EXPECT_EQ(fabs(a), fabs(-4.0f));
	EXPECT_EQ(pow(a, 2), pow(-4.0f, 2));
	EXPECT_EQ(exp(a), exp(-4.0f));

	a = 3.0f;
	EXPECT_EQ(log(a), log(3.0f));

	float c = 1;
	FixedPoint<2, 2> b(1.25);
	c += b;
	EXPECT_EQ(c, 2.25);

	FixedPoint<8, 8> d(259);
	EXPECT_LT(d.toFloat(), 128);
	EXPECT_GT(d.toFloat(), 127);

	FixedPoint<8, 4> e(-259);
	EXPECT_EQ(e.toFloat(), -128);
}