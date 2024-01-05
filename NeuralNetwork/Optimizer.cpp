//
// Created by mat on 31/12/23.
//

#include "Optimizer.h"

Optimizer::Optimizer(Optimizers type) : type(type), numParams(0)
{

}

SGDOptimizer::SGDOptimizer(const float learningRate) : Optimizer(SGD), learningRate(learningRate)
{

}

void
SGDOptimizer::UpdateParameters(Matrix& parameters, const Matrix& gradients, const int offset [[maybe_unused]])
{
	for (int i = 0; i < gradients.matrixSize; ++i)
		parameters.data[i] -= learningRate * gradients.data[i];
}

Optimizer* SGDOptimizer::Copy() const
{
	SGDOptimizer* copy = new SGDOptimizer(learningRate);

	return copy;
}

void SGDOptimizer::Compile(const int numParams [[maybe_unused]])
{
}

SGDOptimizer::SGDOptimizer(FILE* file) : Optimizer(SGD), learningRate(0)
{
	if (fread(&learningRate, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Failed to read learning rate from file");
}

void SGDOptimizer::SaveParametersToFile(FILE* file [[maybe_unused]])
{

}

void SGDOptimizer::LoadParametersFromFile(FILE* file [[maybe_unused]])
{

}

AdamOptimizer::AdamOptimizer() : Optimizer(Adam)
{
	alpha = .01f;
	beta1 = .9f;
	beta2 = .999f;
	gamma = 10e-8f;
	adjBeta1 = beta1;
	adjBeta2 = beta2;
	momentum1 = nullptr;
	momentum2 = nullptr;
	biasCorrectedMomentum1 = nullptr;
	biasCorrectedMomentum2 = nullptr;
}

void AdamOptimizer::UpdateParameters(Matrix& parameters, const Matrix& gradients, const int offset)
{
	float* _momentum1 = momentum1 + offset;
	float* _momentum2 = momentum2 + offset;
	float* _biasCorrectedMomentum1 = biasCorrectedMomentum1 + offset;
	float* _biasCorrectedMomentum2 = biasCorrectedMomentum2 + offset;
	for (int i = 0; i < gradients.matrixSize; ++i)
	{
		_momentum1[i] = beta1 * _momentum1[i] + (1 - beta1) * gradients.data[i];
		_momentum2[i] = beta2 * _momentum2[i] + (1 - beta2) * gradients.data[i] * gradients.data[i];

		_biasCorrectedMomentum1[i] = _momentum1[i] / (1 - adjBeta1);
		_biasCorrectedMomentum2[i] = _momentum2[i] / (1 - adjBeta2);

		parameters.data[i] -= alpha * _biasCorrectedMomentum1[i] / (std::sqrt(_biasCorrectedMomentum2[i]) + gamma);
	}

	adjBeta1 *= beta1;
	adjBeta2 *= beta2;
}

AdamOptimizer::~AdamOptimizer()
{
	delete[] momentum1;
	delete[] momentum2;
	delete[] biasCorrectedMomentum1;
	delete[] biasCorrectedMomentum2;
}

Optimizer* AdamOptimizer::Copy() const
{
	AdamOptimizer* copy = new AdamOptimizer();

	// Copy scalars
	copy->alpha = alpha;
	copy->beta1 = beta1;
	copy->beta2 = beta2;
	copy->gamma = gamma;
	copy->adjBeta1 = adjBeta1;
	copy->adjBeta2 = adjBeta2;

	// Copy arrays
	std::copy(momentum1, momentum1 + numParams, copy->momentum1);
	std::copy(momentum2, momentum2 + numParams, copy->momentum2);
	std::copy(biasCorrectedMomentum1, biasCorrectedMomentum1 + numParams, copy->biasCorrectedMomentum1);
	std::copy(biasCorrectedMomentum2, biasCorrectedMomentum2 + numParams, copy->biasCorrectedMomentum2);

	return copy;
}

void AdamOptimizer::Compile(const int numParams)
{
	momentum1 = new float[numParams]();
	momentum2 = new float[numParams]();
	biasCorrectedMomentum1 = new float[numParams]();
	biasCorrectedMomentum2 = new float[numParams]();
}

AdamOptimizer::AdamOptimizer(FILE* file) : Optimizer(Adam)
{
	alpha = beta1 = beta2 = gamma = adjBeta1 = adjBeta2 = 0;
	momentum1 = momentum2 = biasCorrectedMomentum1 = biasCorrectedMomentum2 = nullptr;
}

void AdamOptimizer::SaveParametersToFile(FILE* file)
{
	fwrite(&alpha, sizeof(float), 1, file);
	fwrite(&beta1, sizeof(float), 1, file);
	fwrite(&beta2, sizeof(float), 1, file);
	fwrite(&gamma, sizeof(float), 1, file);
	fwrite(&adjBeta1, sizeof(float), 1, file);
	fwrite(&adjBeta2, sizeof(float), 1, file);
	fwrite(momentum1, sizeof(float), numParams, file);
	fwrite(momentum2, sizeof(float), numParams, file);
	fwrite(biasCorrectedMomentum1, sizeof(float), numParams, file);
	fwrite(biasCorrectedMomentum2, sizeof(float), numParams, file);
}

void AdamOptimizer::LoadParametersFromFile(FILE* file)
{
	if (fread(&alpha, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Failed to read alpha from file");
	if (fread(&beta1, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Failed to read beta1 from file");
	if (fread(&beta2, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Failed to read beta2 from file");
	if (fread(&gamma, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Failed to read gamma from file");
	if (fread(&adjBeta1, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Failed to read adjBeta1 from file");
	if (fread(&adjBeta2, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Failed to read adjBeta2 from file");
	if (fread(momentum1, sizeof(float), numParams, file) != numParams)
		throw std::runtime_error("Failed to read momentum1 from file");
	if (fread(momentum2, sizeof(float), numParams, file) != numParams)
		throw std::runtime_error("Failed to read momentum2 from file");
	if (fread(biasCorrectedMomentum1, sizeof(float), numParams, file) != numParams)
		throw std::runtime_error("Failed to read biasCorrectedMomentum1 from file");
	if (fread(biasCorrectedMomentum2, sizeof(float), numParams, file) != numParams)
		throw std::runtime_error("Failed to read biasCorrectedMomentum2 from file");
}
