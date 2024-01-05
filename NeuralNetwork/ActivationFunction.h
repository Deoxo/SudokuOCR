//
// Created by mat on 31/12/23.
//

#ifndef SUDOKUOCR_ACTIVATIONFUNCTION_H
#define SUDOKUOCR_ACTIVATIONFUNCTION_H

#include "../Matrix.h"
#include "NetworkUtils.h"

class ActivationFunction
{
public:
	ActivationFunctions type;

	explicit ActivationFunction(const ActivationFunctions& type);

	virtual ~ActivationFunction() = default;

	[[nodiscard]] virtual ActivationFunction* Copy() const = 0;

	virtual void Prime(const Matrix& input, Matrix& output) = 0;

	virtual void Function(const Matrix& input, Matrix& output) = 0;
};

class NoActivation : public ActivationFunction
{
public:
	NoActivation();

	[[nodiscard]] ActivationFunction* Copy() const override;

	void Function(const Matrix& input, Matrix& output) override;

	void Prime(const Matrix& input, Matrix& output) override;
};

class SigmoidActivation : public ActivationFunction
{
public:
	explicit SigmoidActivation() : ActivationFunction(Sigmoid)
	{}

	void Function(const Matrix& input, Matrix& output) override;

	void Prime(const Matrix& input, Matrix& output) override;

	[[nodiscard]] ActivationFunction* Copy() const override;
};

class ReLUActivation : public ActivationFunction
{
public:
	explicit ReLUActivation() : ActivationFunction(ReLU)
	{}

	void Function(const Matrix& input, Matrix& output) override;

	void Prime(const Matrix& input, Matrix& output) override;

	[[nodiscard]] ActivationFunction* Copy() const override;
};

class SoftmaxActivation : public ActivationFunction
{
public:
	explicit SoftmaxActivation() : ActivationFunction(Softmax)
	{}

	void Function(const Matrix& input, Matrix& output) override;

	void Prime(const Matrix& input, Matrix& output) override;

	[[nodiscard]] ActivationFunction* Copy() const override;
};

class TanhActivation : public ActivationFunction
{
public:
	explicit TanhActivation() : ActivationFunction(Tanh)
	{}

	void Function(const Matrix& input, Matrix& output) override;

	void Prime(const Matrix& input, Matrix& output) override;

	[[nodiscard]] ActivationFunction* Copy() const override;
};

class LeakyReLUActivation : public ActivationFunction
{
public:
	explicit LeakyReLUActivation() : ActivationFunction(LeakyReLU)
	{}

	void Function(const Matrix& input, Matrix& output) override;

	void Prime(const Matrix& input, Matrix& output) override;

	[[nodiscard]] ActivationFunction* Copy() const override;
};


#endif //SUDOKUOCR_ACTIVATIONFUNCTION_H
