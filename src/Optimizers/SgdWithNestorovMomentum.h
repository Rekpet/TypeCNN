/*
 * @author Petr Rek
 * @project CNN library
 * @brief Stochastic gradient descent optimizer with nestorov momentum
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef SGD_WITH_NESTOROV_MOMENTUM_H
#define SGD_WITH_NESTOROV_MOMENTUM_H

#include "src/Optimizers/IOptimizer.h"

/* 
 * @brief Stochastic gradient descent optimizer with momentum
 */
class SgdWithNestorovMomentum : public IOptimizer
{

public:

	SgdWithNestorovMomentum();

	virtual void initialize(const unsigned vectorSize, const unsigned vectorsNum,
		const Dimensions & matrixSize, const unsigned matricesNum) override;

	virtual void updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize) override;

	virtual void updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize) override;

	virtual void updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize) override;

	virtual std::unique_ptr<IOptimizer> clone() const override;

	/// Learning rate in base class

	/// Weight decay in base class

	/// Momentum
	BackwardType momentum;

private:

	/// Previous gradients to be used with set momentum
	std::vector<Image<BackwardType>> prevMatrixGradientsVector;

	/// Previous gradients to be used with set momentum
	std::vector<std::vector<BackwardType>> prevVectorGradientsVector;

};

#endif