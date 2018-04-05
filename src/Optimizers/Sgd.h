/*
 * @author Petr Rek
 * @project CNN library
 * @brief Stochastic gradient descent optimizer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef SGD_H
#define SGD_H

#include "src/Optimizers/IOptimizer.h"

/* 
 * @brief Stochastic gradient descent optimizer
 */
class Sgd : public IOptimizer
{

public:

	Sgd();

	virtual void initialize(const unsigned vectorSize = 0, const unsigned vectorsNum = 0, 
		const Dimensions & matrixSize = Dimensions{0, 0, 0}, const unsigned matricesNum = 0) override;

	virtual void updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize) override;

	virtual void updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize) override;

	virtual void updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize) override;

	virtual std::unique_ptr<IOptimizer> clone() const override;

	/// Learning rate in base class

	/// Weight decay in base class

};

#endif