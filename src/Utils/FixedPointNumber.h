/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Fixed point number representation
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef FIXED_POINT_H
#define FIXED_POINT_H

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
struct FixedPoint
{

public:

	/*
	 * @brief Constructs new fixed point number as zero
	 */
	FixedPoint() 
		: m(0)
	{ 
	}

	/*
	 * @brief Constructs new fixed point number from float
	 */
	FixedPoint(float d) 
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
		std::bitset<32> b(m);
		return b.to_string<char, std::char_traits<char>, std::allocator<char> >();
	}

	// Arithmetic operations
	FixedPoint & operator+=(const FixedPoint & x) { m += x.m; applyMask(m); return *this; }
	FixedPoint & operator-=(const FixedPoint & x) { m -= x.m; applyMask(m); return *this; }
	FixedPoint & operator*=(const FixedPoint & x) { m *= x.m; m >>= E; applyMask(m); return *this; }
	FixedPoint & operator/=(const FixedPoint & x) { float tmp = static_cast<float>(m); tmp /= x.m; m = static_cast<int32_t>(tmp * factor); applyMask(m); return *this; }
	FixedPoint & operator*=(int32_t x) { m *= x; applyMask(m); return *this; }
	FixedPoint & operator/=(int32_t x) { m /= x; applyMask(m); return *this; }

	// Friend functions
	friend FixedPoint operator+(FixedPoint x, const FixedPoint & y) { return x += y; }
	friend FixedPoint operator-(FixedPoint x, const FixedPoint & y) { return x -= y; }
	friend FixedPoint operator*(FixedPoint x, const FixedPoint & y) { return x *= y; }
	friend FixedPoint operator/(FixedPoint x, const FixedPoint & y) { return x /= y; }
	friend FixedPoint operator-(FixedPoint x) { x.m = -x.m; applyMask(x.m); return x; }
	
	// Comparison operators
	friend bool operator==(const FixedPoint & x, const FixedPoint & y) { return x.m == y.m; }
	friend bool operator!=(const FixedPoint & x, const FixedPoint & y) { return x.m != y.m; }
	friend bool operator>(const FixedPoint & x, const FixedPoint & y) { return x.m > y.m; }
	friend bool operator<(const FixedPoint & x, const FixedPoint & y) { return x.m < y.m; }
	friend bool operator>=(const FixedPoint & x, const FixedPoint & y) { return x.m >= y.m; }
	friend bool operator<=(const FixedPoint & x, const FixedPoint & y) { return x.m <= y.m; }

	// Stream operators
	friend std::ostream & operator<<(std::ostream & os, const FixedPoint & x) { os << x.toFloat(); return os; }
	friend std::istream & operator>>(std::istream & is, FixedPoint & x) { float tmp; is >> tmp; x = tmp; return is; }
	friend std::string to_string(const FixedPoint & x) { return std::to_string(x.toFloat()); }

public:

	/*
	 * @brief Returns minimum value
	 */
	static FixedPoint getMinimumValue()
	{
		FixedPoint x;
		x.m = negMask;
		return x;
	}

	/*
	 * @brief Returns maximum value
	 */
	static FixedPoint getMaximumValue()
	{
		FixedPoint x;
		x.m = mask;
		return x;
	}

	/*
	 * @brief Returns maximum value
	 */
	static FixedPoint getEpsilonValue()
	{
		FixedPoint x = 0;
		x.m += 1;
		return x;
	}

private:

	/*
	 * @brief Applies mask to limit number of integer bits
	 */
	static void applyMask(int32_t & val)
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
	int32_t m;

	/// Largest possible in our representation
	static const int32_t mask = 0b11111111111111111111111111111111 >> (32 - E - F + 1);

	/// Lowest possible in our representation
	static const int32_t negMask = 0b11111111111111111111111111111111 << (E + F - 1);

	/// Factor to convert to float
	static const int32_t factor = 1 << E;

};

#endif