//
// Created by mat on 20/09/23.
//

#ifndef S3PROJECT_NETWORK_H
#define S3PROJECT_NETWORK_H

#include "../Matrix.h"
#include "NetworkUtils.h"
#include "Layer.h"
#include "CostFunction.h"

struct BackpropagationThreadsArgs;

class NeuralNetwork
{
public:
	NeuralNetwork(Optimizers optimizer, CostFunctions costFunction, float learningRate);

	~NeuralNetwork();

	void Compile();

	void AddLayer(Layer* layer);

	static NeuralNetwork* LoadFromFile(const QString& path);

	void SaveToFile(const QString& path) const;

	CostFunction* costFunction;

	[[nodiscard]] int Predict(const Matrix& input);

	[[nodiscard]] float
	ComputeAccuracy(const Matrix** inputs, const Matrix** outputs, int numSets, bool training);

	void Train(const Matrix** trainingInputs, const Matrix** trainingOutputs, int numInputs, int numberOfEpochs,
			   int batchSize, int numThreads);

private:
	[[nodiscard]] NeuralNetwork* Copy() const;

	void CopyValuesTo(NeuralNetwork& destination) const;

	void Backpropagate(const Matrix& input, const Matrix& target);

	void FeedForward(const Matrix& input, bool training);

	static void* BackpropagationThread(void* arg);

	[[nodiscard]] static CostFunction* CreateCostFunction(CostFunctions type);

	void PrintProgress(const Matrix** trainingInputs, const Matrix** trainingOutputs, int numInputs, int epoch,
					   int numberOfEpochs);

	BackpropagationThreadsArgs*
	CreateBackpropagationThreadsArgs(int numAuxThreads, int numInputsPerThread,
									 const Matrix** trainingOutputs, const Matrix** trainingInputs) const;

	void UpdateAuxNetworks(BackpropagationThreadsArgs* args, int numAuxThreads, int batchOffset,
						   int numInputsPerThread) const;

	static void BackPropagateInAuxThreads(pthread_t* threads, BackpropagationThreadsArgs* args, int numAuxThreads);

	void BackPropagateInMainThread(const Matrix** trainingInputs, const Matrix** trainingOutputs, int offset,
								   int numInputsInMainThread);

	static void WaitAuxThreads(pthread_t* threads, int numAuxThreads);

	void UpdateParameters(int numAuxThreads, BackpropagationThreadsArgs* threadsArgs, float batchScaleFactor);

	Optimizers optimizer;
	float learningRate;
	bool compiled = false;
	Matrix* costDerivative;

	std::vector<Layer*> layers;
	int numLayers;
};

typedef struct BackpropagationThreadsArgs
{
	NeuralNetwork* network;
	int numInputs, offset;
	const Matrix** trainingInputs, ** trainingOutputs;
} BackpropagationThreadsArgs;

#endif //S3PROJECT_NETWORK_H