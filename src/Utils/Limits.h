/*
* @author Petr Rek
* @project CNN Library
* @brief Gets min/max value for given data type
*/

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LIMITS_H
#define LIMITS_H

#include "src/CompileSettings.h"

#include <limits>

/*
 * @brief Wraps around limits for selected types (in case we use FixedPoint)
 */
namespace Limits
{

	/*
	 * @brief Custom types
	 */
	template<class E>
	inline E getMaximumValue()
	{
		return E::getMaximumValue();
	}
	template<class E>
	inline E getMinimumValue()
	{
		return E::getMinimumValue();
	}
	template<class E>
	inline E getEpsilonValue()
	{
		return E::getEpsilonValue();
	}

	/*
	 * @brief Standard types (add more if necessary)
	 */
	#define EXPAND_TYPE(TYPE) \
	template<> \
	inline TYPE getMaximumValue<TYPE>() { return std::numeric_limits<TYPE>::max(); } \
	template<> \
	inline TYPE getMinimumValue<TYPE>() { return std::numeric_limits<TYPE>::min(); } \
	template<> \
	inline TYPE getEpsilonValue<TYPE>() { return std::numeric_limits<TYPE>::epsilon(); }

	EXPAND_TYPE(float);
	EXPAND_TYPE(double);
	EXPAND_TYPE(int);
	EXPAND_TYPE(unsigned);
	EXPAND_TYPE(char);

}

#endif