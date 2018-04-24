/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Training settings
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef TRAINING_SETTINGS_H
#define TRAINING_SETTINGS_H

/*
 * @brief Type of task (C)NN is solving (adjusts return types and output messages)
 */
enum class TaskType
{
	Classification,
	Regression
};

/*
 * Specifies loss function that computes how incorrect current output is
 */
enum class LossFunctionType
{
	MeanSquaredError, // for regression
	CrossEntropy, // for multiclass classification
	BinaryCrossEntropy // for binary classification
};

/*
 * @brief Defines available training optimizers
 */
enum class OptimizerType
{
	Adagrad,
	Adam,
	Sgd,
	SgdWithMomentum,
	SgdWithNestorovMomentum
};

/*
 * @brief Settings that are used by training algorithm
 */
struct TrainingSettings
{

	/// Number of training epochs
	unsigned epochs = 10;

	/// Each X epochs, current error will be written to std::cout (if output is enabled)
	unsigned epochOutputRate = 1;

	/// Periodic output of average error (0 == none)
	unsigned errorOutputRate = 0;

	/// Validation before and after each epoch
	bool periodicValidation = false;

	/// Shuffle training data before each epoch begins
	bool shuffle = false;

	/// Batch size
	/// 1	                == Stochastic (online) gradient descent
	/// 1 < n < Data size   == Minibatch gradient descent
	/// Data size           == Batch gradient descent
	unsigned batchSize = 1;

};

#endif