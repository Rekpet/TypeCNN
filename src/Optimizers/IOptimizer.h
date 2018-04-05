/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Interface for optimizers
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef OPTIMIZER_INTERFACE_H
#define OPTIMIZER_INTERFACE_H

#include "src/CompileSettings.h"
#include "src/Image.h"
#include "src/TrainingSettings.h"

/*
 * @brief Interface all optimizers need to follow
 * @brief Great overview here: http://ruder.io/optimizing-gradient-descent/index.html
 */
class IOptimizer
{

public:

	virtual ~IOptimizer() {};

	/*
	 * @brief Initializes optimizer with training settings and also creates place to store temporary values (voluntary)
	 * 
	 * @param vectorSize    Size of vectors to hold temporary values
	 * @param vectorsNum    Number of vectors with declared size
	 * @param matrixSize    Size of matrices to hold temporary values
	 * @param matricesNum   Number of matrices with declared size
	 */
	virtual void initialize(const unsigned vectorSize, const unsigned vectorsNum, 
		const Dimensions & matrixSize, const unsigned matricesNum) = 0;

	/*
	 * @brief Updates weights of single matrix (e.g. FC layer weights)
	 *
	 * @param weight        Matrix containing weights
	 * @param delta         Values used to update weights
	 * @param batchSize     Batch size (to compute average)
	 */
	virtual void updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize) = 0;

	/*
	 * @brief Updates weights of single vector (e.g. convolutional bias)
	 *
	 * @param weight        Vector containing weights
	 * @param delta         Values used to update weights
	 * @param batchSize     Batch size (to compute average)
	 */
	virtual void updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize) = 0;

	/*
	 * @brief Updates weights of matrices (e.g. convolutional filters)
	 *
	 * @param weight       Matrices containing weights
	 * @param delta        Values used to update weights
	 * @param batchSize    Batch size (to compute average)
	 */
	virtual void updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize) = 0;

	/*
	* @brief Clones uninitialized optimizer object
	*/
	virtual std::unique_ptr<IOptimizer> clone() const = 0;

	/// Learning coefficient
	BackwardType learningRate;

	/// Weight decay
	BackwardType weightDecay;

};

#endif
