/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Interface for layers
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LAYER_INTERFACE_H
#define LAYER_INTERFACE_H

#include "../Image.h"
#include "../TrainingSettings.h"
#include "../Optimizers/IOptimizer.h"

#include <istream>
#include <ostream>

/*
 * @brief Generic exception thrown by layers
 */
class CNNException : public std::runtime_error
{

public:

	explicit CNNException(const std::string & msg)
		: std::runtime_error(msg.c_str()) {};

};

/*
 * @brief Exception thrown if input to layer has different dimensions than declared during initialization
 */
class InputImageDoesNotHaveCorrectDimensions : public CNNException 
{ 

	using CNNException::CNNException; 

};

/*
 * @brief Interface all layers need to follow in order for them to be used in CNN interface
 */
template <class _ForwardType, class _WeightType>
class ILayer
{

public:

	ILayer() = default;
	~ILayer() = default;

	/*
	 * @brief Forward propagates an input matrix
	 *
	 * @param in Input matrix
	 * @param out Output matrix (external or inside layer)
	 *
	 * @throws InputImageDoesNotHaveCorrectDimensions if input dimensions do not correspond to the ones declared during initialization
	 */
	virtual void forwardPropagation(const Image<_ForwardType> & in, Image<_ForwardType> & out) = 0;

	/*
	 * @brief Backward propagation to compute gradients and update learnable parameters
	 *
	 * @param in                 Original input during forward propagation
	 * @param out                Original output during backward propagation (external or inside layer)
	 * @param inGradients        Input gradients used to update learnable parameters in this layer
	 * @param outGradients       Output gradients used to update learnable parameters in previous layer (external or inside layer)
	 * @param trainingSettings   Settings for training (learning coefficient, batch size etc.)
	 */
	virtual void backwardPropagation(const Image<_ForwardType> & in, const Image<_ForwardType> & out, const Image<BackwardType> & inGradients,
		                             Image<BackwardType> & outGradients, const TrainingSettings & trainingSettings) = 0;

	/*
	 * @brief Returns expected input dimensions
	 */
	virtual Dimensions getInputSize() const = 0;

	/*
	 * @brief Returns output dimensions
	 */
	virtual Dimensions getOutputSize() const = 0;

	/*
	 * @brief Returns a reference to this layers output (can be used both for reading and writing)
	 */
	virtual Image<_ForwardType> & getOutput() = 0;

	/*
	 * @brief Returns a reference to this layers gradient output (can be used both for reading and writing)
	 */
	virtual Image<BackwardType> & getGradientOutput() = 0;

	/*
	 * @brief Initializes optimizer (needed only for layers with learnable parameters)
	 *
	 * @param trainingSettings   Training settings that initialize optimizer (e.g. momentum, weight decay etc.)
	 */
	virtual void initializeOptimizer() {};

public:

	/*
	 * @brief Sets an optimizer
	 *
	 * @param  opt   Optimizer to be used
	 */
	void setOptimizer(const std::shared_ptr<IOptimizer> opt)
	{
		optimizer = opt->clone();
	}

public:

	/// This layer should be used only when learning, not during predictions
	bool useOnlyWhenLearning = false;

protected:

	/// Optimizer pointer
	std::unique_ptr<IOptimizer> optimizer;

};

#endif
