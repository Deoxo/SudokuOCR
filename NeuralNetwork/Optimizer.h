//
// Created by mat on 31/12/23.
//

#ifndef SUDOKUOCR_OPTIMIZER_H
#define SUDOKUOCR_OPTIMIZER_H

#include "Tools/Matrix.h"
#include "NetworkUtils.h"

class Optimizer
{
public:
	Optimizers type;

	explicit Optimizer(Optimizers type);

	virtual void Compile(int numParams) = 0;

	virtual ~Optimizer() = default;

	[[nodiscard]] virtual Optimizer* Copy() const = 0;

	virtual void UpdateParameters(Matrix& parameters, const Matrix& gradients, int offset) = 0;

	virtual void SaveParametersToFile(FILE* file) = 0;

	virtual void LoadParametersFromFile(FILE* file) = 0;

protected:
	int numParams;
};

class SGDOptimizer : public Optimizer
{
public:
	explicit SGDOptimizer(float learningRate);

	explicit SGDOptimizer(FILE* file);

	void UpdateParameters(Matrix& parameters, const Matrix& gradients, int offset) override;

	[[nodiscard]] Optimizer* Copy() const override;

	void Compile(int numParams) override;

	void SaveParametersToFile(FILE* file) override;

	void LoadParametersFromFile(FILE* file) override;

private:
	float learningRate;
};

class AdamOptimizer : public Optimizer
{
public:
	AdamOptimizer();

	explicit AdamOptimizer(FILE* file);

	~AdamOptimizer() override;

	void UpdateParameters(Matrix& parameters, const Matrix& gradients, int offset) override;

	[[nodiscard]] Optimizer* Copy() const override;

	void Compile(int numParams) override;

	void SaveParametersToFile(FILE* file) override;

	void LoadParametersFromFile(FILE* file) override;

private:
	float alpha;
	float beta1;
	float beta2;
	float adjBeta1;
	float adjBeta2;
	float gamma;

	float* momentum1;
	float* momentum2;

	float* biasCorrectedMomentum1;
	float* biasCorrectedMomentum2;
};


#endif //SUDOKUOCR_OPTIMIZER_H
