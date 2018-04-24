/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Maps enum classes to strings
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef PERSISTENCE_MAPPER_H
#define PERSISTENCE_MAPPER_H

#include "src/ConvolutionalNeuralNetwork.h"
#include "src/Layers/PoolingLayer.h"
#include "src/Layers/ActivationLayer.h"

 /*
  * @brief Exception thrown if attribute has not been mapped
  */
class AttributeIsNotMapped : public std::runtime_error
{
public:

	AttributeIsNotMapped(const std::string & msg) : std::runtime_error(msg.c_str())
	{
	}
};

/*
 * @brief This module provides mapping functions for enums defined in layers etc. Easier handling in persistency. 
 */
namespace PersistenceMapper
{

/*
 * @brief Finds string equivalent of given enum class item
 */
template <class EC>
std::string getStringForEnumItem(const EC & enumType, const std::vector<std::pair<std::string, EC>> & enumMap)
{
	for (const auto & item : enumMap)
	{
		if (item.second == enumType)
		{
			return item.first;
		}
	}

	throw AttributeIsNotMapped("Cannot find string for given enum item.");
}


/*
 * @brief Finds enum class item equivalent of given string
 */
template <class EC>
EC getEnumItemForString(const std::string & str, const std::vector<std::pair<std::string, EC>> & enumMap)
{
	for (const auto & item : enumMap)
	{
		if (item.first == str)
		{
			return item.second;
		}
	}

	throw AttributeIsNotMapped("Cannot find enum item for given string \"" + str + "\".");
}


/*
 * @brief Activation function mapping
 */
const std::vector<std::pair<std::string, ActivationFunction>> activationFunctionMap =
	{
		{ "sigmoid", ActivationFunction::Sigmoid },
		{ "tanh", ActivationFunction::Tanh },
		{ "relu", ActivationFunction::ReLU },
		{ "leaky_relu", ActivationFunction::LeakyReLU },
		{ "softmax", ActivationFunction::SoftMax }
	};

inline ActivationFunction getActivationFunctionType(const std::string & str)
{
	return getEnumItemForString(str, activationFunctionMap);
}

inline std::string getActivationFunctionString(const ActivationFunction & item)
{
	return getStringForEnumItem(item, activationFunctionMap);
}

inline std::shared_ptr<ActivationLayer<ForwardType, WeightType>> getActivationLayer(const ActivationFunction & type, const Dimensions & dim)
{
	switch (type)
	{
		case ActivationFunction::LeakyReLU:
			return std::make_shared<LeakyReLU>(dim);
		case ActivationFunction::ReLU:
			return std::make_shared<ReLU>(dim);
		case ActivationFunction::Sigmoid:
			return std::make_shared<Sigmoid>(dim);
		case ActivationFunction::SoftMax:
			return std::make_shared<SoftMax>(dim);
		case ActivationFunction::Tanh:
			return std::make_shared<Tanh>(dim);
		case ActivationFunction::None: default:
			return nullptr;
	}
}

/*
 * @brief Pooling operation mapping
 */
const std::vector<std::pair<std::string, PoolingOperation>> poolingOperationMap =
{
	{ "max", PoolingOperation::Max },
	{ "avg", PoolingOperation::Average }
};

inline PoolingOperation getPoolingOperationType(const std::string & str)
{
	return getEnumItemForString(str, poolingOperationMap);
}

inline std::string getPoolingOperationString(const PoolingOperation & item)
{
	return getStringForEnumItem(item, poolingOperationMap);
}


/*
 * @brief Loss function mapping
 */
const std::vector<std::pair<std::string, LossFunctionType>> lossFunctionMap =
{
	{ "MSE", LossFunctionType::MeanSquaredError },
	{ "CE", LossFunctionType::CrossEntropy },
	{ "CEbin", LossFunctionType::BinaryCrossEntropy }
};

inline LossFunctionType getLossFunctionType(const std::string & str)
{
	return getEnumItemForString(str, lossFunctionMap);
}

inline std::string getLossFunctionString(const LossFunctionType & item)
{
	return getStringForEnumItem(item, lossFunctionMap);
}


/*
 * @brief Task type mapping
 */
const std::vector<std::pair<std::string, TaskType>> taskTypeMap =
{
	{ "classification", TaskType::Classification },
	{ "regression", TaskType::Regression }
};

inline TaskType getTaskType(const std::string & str)
{
	return getEnumItemForString(str, taskTypeMap);
}

inline std::string getTaskTypeString(const TaskType & item)
{
	return getStringForEnumItem(item, taskTypeMap);
}


/*
 * @brief Optimizer type mapping
 */
const std::vector<std::pair<std::string, OptimizerType>> optimizerTypeMap =
{
	{ "sgd", OptimizerType::Sgd },
	{ "sgdm", OptimizerType::SgdWithMomentum },
	{ "sgdn", OptimizerType::SgdWithNestorovMomentum },
	{ "adagrad", OptimizerType::Adagrad },
	{ "adam", OptimizerType::Adam }
};

inline OptimizerType getOptimizerType(const std::string & str)
{
	return getEnumItemForString(str, optimizerTypeMap);
}

inline std::string getOptimizerTypeString(const OptimizerType & item)
{
	return getStringForEnumItem(item, optimizerTypeMap);
}

inline std::shared_ptr<IOptimizer> getOptimizerInstance(const OptimizerType & type)
{
	switch (type)
	{
		case OptimizerType::SgdWithMomentum: default:
			return std::make_shared<SgdWithMomentum>();
		case OptimizerType::Sgd:
			return std::make_shared<Sgd>();
		case OptimizerType::Adagrad:
			return std::make_shared<Adagrad>();
		case OptimizerType::Adam:
			return std::make_shared<Adam>();
		case OptimizerType::SgdWithNestorovMomentum:
			return std::make_shared<SgdWithNestorovMomentum>();
	}
}

} // namespace PersistenceMapper

#endif