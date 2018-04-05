/*
 * @author Petr Rek
 * @project CNN library
 * @brief Adagrad optimizer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef ADAGRAD_H
#define ADAGRAD_H

#include "src/Optimizers/IOptimizer.h"

/* 
 * @brief Adagrad optimizer
 */
class Adagrad : public IOptimizer
{

public:

	Adagrad();

	virtual void initialize(const unsigned vectorSize, const unsigned vectorsNum, 
		const Dimensions & matrixSize, const unsigned matricesNum) override;

	virtual void updateWeights(Image<BackwardType> & weight, Image<BackwardType> & delta, const unsigned batchSize) override;

	virtual void updateWeights(std::vector<Image<BackwardType>> & weight, std::vector<Image<BackwardType>> & delta, const unsigned batchSize) override;

	virtual void updateWeights(std::vector<BackwardType> & weight, std::vector<BackwardType> & delta, const unsigned batchSize) override;

	virtual std::unique_ptr<IOptimizer> clone() const override;

	/// Learning rate in base class

	/// Weight decay in base class

	/// Epsilon
	BackwardType epsilon;

private:

	/// Previous squared gradients to be used
	std::vector<Image<BackwardType>> prevMatrixGradientsVector;

	/// Previous squared gradients to be used
	std::vector<std::vector<BackwardType>> prevVectorGradientsVector;

};

#endif