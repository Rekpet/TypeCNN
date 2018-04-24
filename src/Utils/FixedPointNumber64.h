/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Fixed point number representation
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef FIXED_POINT_64_H
#define FIXED_POINT_64_H

#include <bitset>
#include <ostream>
#include <istream>
#include <cmath>
#include <string>

/*
 * Based on https://alikhuram.wordpress.com/2013/05/20/implementing-fixed-point-numbers-in-c/
 * Added new functionalaty and fixed problems
 */

/*
 * @brief Fixed point representation
 *			- F is whole part bits
 *			- E is decimal part bits
 */
template<int F, int E>
struct FixedPoint64
{

public:

	/*
	 * @brief Constructs new fixed point number as zero
	 */
	FixedPoint64() 
		: m(0)
	{ 
	}

	/*
	 * @brief Constructs new fixed point number from float
	 */
	FixedPoint64(float d) 
		: m(static_cast<int>(d * factor))
	{
		applyMask(m); 
	}

	/*
	 * @brief Explicit cast to float
	 */
	float toFloat() const {
		return float(m) / factor;
	}

	/*
	 * @brief Implicit cast to float
	 */
	operator float()
	{
		return toFloat();
	}

	/*
	 * @brief Returns binary value of fixed point number
	 */
	std::string getAsBitString()
	{
		std::bitset<64> b(m);
		return b.to_string<char, std::char_traits<char>, std::allocator<char> >();
	}

	// Arithmetic operations
	FixedPoint64 & operator+=(const FixedPoint64 & x) { m += x.m; applyMask(m); return *this; }
	FixedPoint64 & operator-=(const FixedPoint64 & x) { m -= x.m; applyMask(m); return *this; }
	FixedPoint64 & operator*=(const FixedPoint64 & x) { m *= x.m; m >>= E; applyMask(m); return *this; }
	FixedPoint64 & operator/=(const FixedPoint64 & x) { float tmp = static_cast<float>(m); tmp /= x.m; m = static_cast<int32_t>(tmp * factor); applyMask(m); return *this; }
	FixedPoint64 & operator*=(int32_t x) { m *= x; applyMask(m); return *this; }
	FixedPoint64 & operator/=(int32_t x) { m /= x; applyMask(m); return *this; }

	// Friend functions
	friend FixedPoint64 operator+(FixedPoint64 x, const FixedPoint64 & y) { return x += y; }
	friend FixedPoint64 operator-(FixedPoint64 x, const FixedPoint64 & y) { return x -= y; }
	friend FixedPoint64 operator*(FixedPoint64 x, const FixedPoint64 & y) { return x *= y; }
	friend FixedPoint64 operator/(FixedPoint64 x, const FixedPoint64 & y) { return x /= y; }
	friend FixedPoint64 operator-(FixedPoint64 x) { x.m = -x.m; applyMask(x.m); return x; }
	
	// Comparison operators
	friend bool operator==(const FixedPoint64 & x, const FixedPoint64 & y) { return x.m == y.m; }
	friend bool operator!=(const FixedPoint64 & x, const FixedPoint64 & y) { return x.m != y.m; }
	friend bool operator>(const FixedPoint64 & x, const FixedPoint64 & y) { return x.m > y.m; }
	friend bool operator<(const FixedPoint64 & x, const FixedPoint64 & y) { return x.m < y.m; }
	friend bool operator>=(const FixedPoint64 & x, const FixedPoint64 & y) { return x.m >= y.m; }
	friend bool operator<=(const FixedPoint64 & x, const FixedPoint64 & y) { return x.m <= y.m; }

	// Stream operators
	friend std::ostream & operator<<(std::ostream & os, const FixedPoint64 & x) { os << x.toFloat(); return os; }
	friend std::istream & operator>>(std::istream & is, FixedPoint64 & x) { float tmp; is >> tmp; x = tmp; return is; }
	friend std::string to_string(const FixedPoint64 & x) { return std::to_string(x.toFloat()); }

public:

	/*
	 * @brief Returns minimum value
	 */
	static FixedPoint64 getMinimumValue()
	{
		FixedPoint64 x;
		x.m = negMask;
		return x;
	}

	/*
	 * @brief Returns maximum value
	 */
	static FixedPoint64 getMaximumValue()
	{
		FixedPoint64 x;
		x.m = mask;
		return x;
	}

	/*
	 * @brief Returns maximum value
	 */
	static FixedPoint64 getEpsilonValue()
	{
		FixedPoint64 x = 0;
		x.m += 1;
		return x;
	}

private:

	/*
	 * @brief Applies mask to limit number of integer bits
	 */
	static void applyMask(int64_t & val)
	{
		if (val > mask)
		{
			val = mask;
		}
		else if (val < negMask)
		{
			val = negMask;
		}
	}

private:

	/// Number in fixed point representation
	int64_t m;

	/// Largest possible in our representation
	static const int64_t mask = 0b1111111111111111111111111111111111111111111111111111111111111111 >> (64 - E - F + 1);

	/// Lowest possible in our representation
	static const int64_t negMask = 0b1111111111111111111111111111111111111111111111111111111111111111 << (E + F - 1);

	/// Factor to convert to float
	static const int64_t factor = 1 << E;

};

#endif