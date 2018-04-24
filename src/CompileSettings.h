/*
 * @author Petr Rek
 * @project CNN library
 * @brief Defines compile time settings
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef COMPILE_SETTINGS_H
#define COMPILE_SETTINGS_H

#include "src/Utils/FixedPointNumber.h"
#include "src/Utils/FixedPointNumber64.h"

/*
 * BackwardType is used for back propagation
 * Use only default types (recommended float/double)
 */
#ifdef CNN_BTYPE
	using BackwardType = CNN_BTYPE;
#else
	using BackwardType = float;
#endif

/*
 * ForwardType is the default type used for forward propagation (can be redefined since classes are templated)
 */
#ifdef CNN_FTYPE
	using ForwardType = CNN_FTYPE;
#else
	using ForwardType = float;
#endif

/*
 * WeightType is defines how weights are used during forward propagation and than saved to persistence
 */
#ifdef CNN_WTYPE
	using WeightType = CNN_WTYPE;
#else
	using WeightType = float;
#endif

#endif
