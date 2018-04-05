/*
 * @author Petr Rek
 * @project CNN library
 * @brief Adam optimizer
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef ADAM_H
#define ADAM_H

#include "src/Optimizers/IOptimizer.h"

/* 
 * @brief Adam optimizer
 */
class Adam : public IOptimizer
{

public:

	Adam();

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

	/// Adam coefficients (default values from authors)
	BackwardType b1;
	BackwardType b2 ;
	BackwardType b1t;
	BackwardType b2t;

private:

	/// Previous gradients to be used
	std::vector<Image<BackwardType>> prevMatrixGradientsVector;

	/// Previous gradients to be used
	std::vector<std::vector<BackwardType>> prevVectorGradientsVector;

	/// Previous squared gradients to be used
	std::vector<Image<BackwardType>> prevMatrixSquaredGradientsVector;

	/// Previous squared gradients to be used
	std::vector<std::vector<BackwardType>> prevVectorSquaredGradientsVector;

};

#endif