//
// Created by mat on 31/12/23.
//

#ifndef SUDOKUOCR_COSTFUNCTION_H
#define SUDOKUOCR_COSTFUNCTION_H

#include "../Matrix.h"
#include "NetworkUtils.h"

class CostFunction
{
public:
	CostFunctions type;

	explicit CostFunction(const CostFunctions& type);

	virtual ~CostFunction() = default;

	virtual float Function(const Matrix& m, const Matrix& target) = 0;

	virtual void Prime(const Matrix& m, const Matrix& target, Matrix& output) = 0;

	[[nodiscard]] virtual CostFunction* Copy() const = 0;
};

class MSE_Cost : public CostFunction
{
public:
	MSE_Cost();

	float Function(const Matrix& m, const Matrix& target) override;

	void Prime(const Matrix& m, const Matrix& target, Matrix& output) override;

	[[nodiscard]] CostFunction* Copy() const override;
};

class CrossEntropyCost : public CostFunction
{
public:
	CrossEntropyCost();

	float Function(const Matrix& m, const Matrix& target) override;

	void Prime(const Matrix& m, const Matrix& target, Matrix& output) override;

	[[nodiscard]] CostFunction* Copy() const override;
};


#endif //SUDOKUOCR_COSTFUNCTION_H
