//
// Created by mat on 20/09/23.
//

#include "Layer.h"
#include "Network.h"
#include "NetworkUtils.h"
#include "DatasetManager.h"
#include "../Tools/Settings.h"
#include <iostream>
#include <unistd.h>

#if MULTITHREAD

#include <pthread.h>
#endif

void* NeuralNetwork::BackpropagationThread(void* arg)
{
	BackpropagationThreadsArgs* args = (BackpropagationThreadsArgs*) arg;
	for (int inputInd = 0; inputInd < args->numInputs; inputInd++)
		args->network->Backpropagate(*args->trainingInputs[args->offset + inputInd],
									 *args->trainingOutputs[args->offset + inputInd]);

	return nullptr;
}

float
NeuralNetwork::ComputeAccuracy(const Matrix** inputs, const Matrix** outputs, const int numSets, const bool training)
{
	int correct = 0;
	for (int i = 0; i < numSets; ++i)
	{
		FeedForward(*inputs[i], training);
		Matrix* output = layers[numLayers - 1]->output;
		int maxIndex = 0;
		for (int j = 0; j < output->matrixSize; ++j)
			if (output->data[j] > output->data[maxIndex])
				maxIndex = j;
		if (outputs[i]->data[maxIndex] == 1)
			correct++;
	}

	return (float) correct / (float) numSets * 100.0f;
}

void NeuralNetwork::PrintProgress(const Matrix** trainingInputs, const Matrix** trainingOutputs, const int numInputs,
								  const int epoch, const int numberOfEpochs)
{
	const float accuracy = ComputeAccuracy(trainingInputs, trainingOutputs, numInputs / 100, true);
	printf("\r%sEpoch %s%i%s / %i: ", BLUE, YELLOW, epoch, BLUE, numberOfEpochs);
	printf("%sAccuracy: %s%.2f%s%%%s", BLUE, YELLOW, accuracy, BLUE, RESET);
	fflush(stdout);
}

BackpropagationThreadsArgs*
NeuralNetwork::CreateBackpropagationThreadsArgs(const int numAuxThreads, const int numInputsPerThread,
												const Matrix** trainingOutputs, const Matrix** trainingInputs) const
{
	BackpropagationThreadsArgs* threadsArgs = new BackpropagationThreadsArgs[numAuxThreads];
	for (int i = 0; i < numAuxThreads; ++i)
	{
		BackpropagationThreadsArgs* args = threadsArgs + i;
		args->network = Copy();
		args->numInputs = numInputsPerThread;
		args->trainingInputs = trainingInputs;
		args->trainingOutputs = trainingOutputs;
	}

	return threadsArgs;
}

void NeuralNetwork::Train(const Matrix** trainingInputs, const Matrix** trainingOutputs, const int numInputs,
						  const int numberOfEpochs, const int batchSize, const int numThreads)
{
	if (!compiled)
		throw std::runtime_error("Network not compiled");

	// Verify parameters
	if (batchSize > numInputs)
		throw std::runtime_error("batchSize must be smaller than the number of inputs");
	if (numInputs % batchSize)
		throw std::runtime_error("the number of inputs must be a multiple of batch size");
	if (numThreads > batchSize)
		throw std::runtime_error("batchSize must be greater than numThreads");

	const int numBatches = numInputs / batchSize;
	const int numInputsPerThread = batchSize / numThreads;
	const int numAuxThreads = numThreads - 1;
	const int numInputsInMainThread = batchSize - numInputsPerThread * numAuxThreads;
	const float batchScaleFactor = 1.f / (float) batchSize;

	// Create threads
	pthread_t* threads = new pthread_t[numAuxThreads];
	BackpropagationThreadsArgs* threadsArgs = CreateBackpropagationThreadsArgs(numAuxThreads, numInputsPerThread,
																			   trainingOutputs, trainingInputs);

	// Train
	for (int epoch = 0; epoch < numberOfEpochs; epoch++)
	{
		for (int b = 0; b < numBatches; ++b)
		{
			// Update auxiliary networks
			const int batchOffset = b * batchSize;
			UpdateAuxNetworks(threadsArgs, numAuxThreads, batchOffset, numInputsPerThread);

			// Backproagate
			BackPropagateInAuxThreads(threads, threadsArgs, numAuxThreads);
			const int offset = batchOffset + numAuxThreads * numInputsPerThread;
			BackPropagateInMainThread(trainingInputs, trainingOutputs, offset, numInputsInMainThread);

			// Wait for threads
			WaitAuxThreads(threads, numAuxThreads);

			// Update parameters (weights and biases)
			UpdateParameters(numAuxThreads, threadsArgs, batchScaleFactor);

			// Print progress
			PrintProgress(trainingInputs, trainingOutputs, numInputs, epoch, numberOfEpochs);
		}
	}
	PrintProgress(trainingInputs, trainingOutputs, numInputs, numberOfEpochs, numberOfEpochs);
	std::cout << std::endl;

	// Free threads and args
	delete[] threads;
	for (int i = 0; i < numAuxThreads; ++i)
		delete threadsArgs[i].network;
	delete[] threadsArgs;
}

void MNIST(const int saveNeuralNetwork)
{
	const int numTrainingSets = 60000;//60000;
	const int numInputsNodes = 784, numHiddenNodes = 100, numOutputsNodes = 10;
	const int batchSize = 100;
	const float learningRate = 1.f;
	const int numberOfEpochs = 1;
	const float dropoutRate = 0.4f;

	Matrix*** MNIST = LoadMnist("./datasets/mnist.csv", numTrainingSets, 0);
	const Matrix** inputs = (const Matrix**) MNIST[0];
	const Matrix** outputs = (const Matrix**) MNIST[1];

	NeuralNetwork* network = new NeuralNetwork(Adam, MSE, learningRate);
	network->AddLayer(new InputLayer(numInputsNodes));
	network->AddLayer(new DropoutLayer(Sigmoid, numHiddenNodes, dropoutRate));
	network->AddLayer(new FCL(Softmax, numOutputsNodes));

	network->Compile();
	network->Train(inputs, outputs, numTrainingSets * .8f, numberOfEpochs, batchSize, NUM_THREADS);

	printf("%sAccuracy on test data: %s%.2f%s%%%s\n", BLUE, YELLOW,
		   network->ComputeAccuracy((const Matrix**) ((Matrix**) inputs + (int) (numTrainingSets * .8f)),
									(const Matrix**) ((Matrix**) outputs + (int) (numTrainingSets * .8f)),
									numTrainingSets * .2f, false), BLUE, RESET);

	if (saveNeuralNetwork)
	{
		network->SaveToFile("./nn.bin");
		printf("Saved neural network\n");
	}

	delete network;
}

/*void MNIST2(int saveNeuralNetwork)
{
    const int numTrainingSets = 10000; //60000;
    const int numInputsNodes = 784, numOutputsNodes = 10;
    const int numLayers = 5;
    const int batchSize = 100;
    const float learningRate = 1.f;

    Matrix*** MNIST = LoadMnist("./datasets/mnist.csv", numTrainingSets, 1);
    const Matrix** inputs = (const Matrix**) MNIST[0];
    const Matrix** outputs = (const Matrix**) MNIST[1];

    const int poolNumNeurons =
            ((28 - POOL_FILTERS_SIZE) / POOL_STRIDE + 1) * ((28 - POOL_FILTERS_SIZE) / POOL_STRIDE + 1);
    const int numberOfEpochs = 5;
    int* numNeuronsPerLayer = new int[]{numInputsNodes, numInputsNodes, poolNumNeurons, 0, numOutputsNodes};
    ActivationFunctions* activationFunctions = new ActivationFunctions[]{None, SigmoidActivation, None, None, SoftmaxActivation};
    LayerTypes* layerTypes = new LayerTypes[]{Input, Conv, MaxPool, Flatten, FC};
    NeuralNetwork* network = CreateNeuralNetwork(numNeuronsPerLayer, numLayers, activationFunctions, layerTypes, learningRate,
                                                 Adam, MSE_Cost);
    TrainNeuralNetwork(network, inputs, outputs, numTrainingSets, numberOfEpochs, batchSize, 1);

    if (saveNeuralNetwork)
    {
        SaveNeuralNetwork(network, "./nn.bin");
        printf("Saved neural network\n");
    }

    delete network;
    delete[] numNeuronsPerLayer;
    delete[] activationFunctions;
    delete[] layerTypes;
}*/

/*void Custom(const int saveNeuralNetwork)
{
    const int numTrainingSets = 34900;//35000;
    const int numInputsNodes = 784, numHiddenNodes = 100, numOutputsNodes = 9;
    const int numLayers = 3;
    const int batchSize = 100;
    const float learningRate = 1.f;

    Matrix*** MNIST = LoadCustom("./datasets/custom", numTrainingSets, 0);
    const Matrix** inputs = (const Matrix**) MNIST[0];
    const Matrix** outputs = (const Matrix**) MNIST[1];


    const int numberOfEpochs = 8;
    int* numNeuronsPerLayer = new int[3]{numInputsNodes, numHiddenNodes, numOutputsNodes};
    ActivationFunctions* activationFunctions = new ActivationFunctions[3]{None, SigmoidActivation, SoftmaxActivation};
    LayerTypes* layerTypes = new LayerTypes[3]{Input, FC, FC};
    NeuralNetwork* network = CreateNeuralNetwork(numNeuronsPerLayer, numLayers, activationFunctions, layerTypes, learningRate,
                                                 Adam, MSE_Cost);
    TrainNeuralNetwork(network, inputs, outputs, numTrainingSets, numberOfEpochs, batchSize, NUM_THREADS);

    Matrix*** testDataset = LoadCustom("./datasets/customTest", 990, 0);
    printf("Accuracy on test data: %.2f%%\n",
           ComputeAccuracy(network, (const Matrix**) testDataset[0], (const Matrix**) testDataset[1], 990));

    if (saveNeuralNetwork)
    {
        SaveNeuralNetwork(network, "./nn.bin");
        printf("Saved neural network\n");
    }

    delete network;
    delete[] numNeuronsPerLayer;
    delete[] activationFunctions;
    delete[] layerTypes;
}*/

void Custom3(const int saveNeuralNetwork)
{
	const int numData = 17658;
	const int numTrainingSets = 17000;//17600;
	const int numInputsNodes = 784, numHiddenNodes = 100, numOutputsNodes = 9;
	const int batchSize = 100;
	const int numberOfEpochs = 15;
	const float dropoutRate = .4f;
	const float learningRate = 1.f;

	Matrix*** MNIST = LoadCustom3("./datasets/custom3", numData, 0);
	const Matrix** inputs = (const Matrix**) MNIST[0];
	const Matrix** outputs = (const Matrix**) MNIST[1];


	NeuralNetwork* network = new NeuralNetwork(Adam, MSE, learningRate);
	network->AddLayer(new InputLayer(numInputsNodes));
	network->AddLayer(new DropoutLayer(Sigmoid, numHiddenNodes, dropoutRate));
	network->AddLayer(new FCL(Sigmoid, numOutputsNodes));
	network->Compile();

	network->Train(inputs, outputs, numTrainingSets, numberOfEpochs, batchSize, NUM_THREADS);

	printf("%sAccuracy on test data: %s%.2f%s%%%s\n", BLUE, YELLOW,
		   network->ComputeAccuracy((const Matrix**) ((Matrix**) inputs + numTrainingSets),
									(const Matrix**) ((Matrix**) outputs + numTrainingSets),
									numData - numTrainingSets, false), BLUE, RESET);

	if (saveNeuralNetwork)
	{
		network->SaveToFile("./nn3.bin");
		printf("Saved neural network\n");
	}

	delete network;
}

/*int main(int argc [[maybe_unused]], char** argv [[maybe_unused]])
{
    //Todo: figure out why using NUM_THREADS instead of 1 results in an accuracy 2% lower
    //Todo: implement CNNs
    //Todo: understand why reLU dies (outputs is 0s)
    //Todo: improve convolutions by removing indexes test from nested for loops and treating edges separately
    //Todo: add proper implementation of flatten layers
    //Todo: Proper save/load system for optimizers and learning rate and special layers parameters
    //Todo: Make proper parameter passing for special layers parameters

    Custom3(1);
    //MNIST(1);
    return 0;

    NeuralNetwork* nn = NeuralNetwork::LoadFromFile("./nn.bin");
    const int numInputs = 60000;
    Matrix*** MNIST = LoadMnist("./datasets/mnist.csv", numInputs, 0);
    const Matrix** inputs = (const Matrix**) MNIST[0];
    const Matrix** outputs = (const Matrix**) MNIST[1];
    const float accuracy = nn->ComputeAccuracy(inputs + (int) (numInputs * .8f), outputs + (int) (numInputs * .8f),
                                               (int) (numInputs * .2f), false);
    printf("Accuracy: %.2f%%\n", accuracy);

    nn->Train(inputs, outputs, (int) (60000 * .8f), 1, 100, NUM_THREADS);
    delete nn;

    return 0;
}*/

NeuralNetwork::~NeuralNetwork()
{
	delete costDerivative;
	for (int i = 1; i < numLayers; ++i)
		delete layers[i];
	delete costFunction;
}

NeuralNetwork* NeuralNetwork::Copy() const
{
	NeuralNetwork* copy = new NeuralNetwork(optimizer, costFunction->type, learningRate);

	for (int i = 0; i < numLayers; ++i)
		copy->AddLayer(layers[i]->Copy());

	copy->Compile();
	CopyValuesTo(*copy);

	return copy;
}

void NeuralNetwork::CopyValuesTo(NeuralNetwork& destination) const
{
	costDerivative->CopyValuesTo(*destination.costDerivative);
	layers[0]->output->CopyValuesTo(*destination.layers[0]->output);
	for (int i = 1; i < numLayers; ++i)
		layers[i]->CopyValuesTo(*destination.layers[i]);
}

void NeuralNetwork::FeedForward(const Matrix& input, const bool training)
{
	Matrix* in = &(Matrix&) input;
	for (int i = 0; i < numLayers; ++i)
	{
		layers[i]->FeedForward(*in, training);
		in = layers[i]->output;
	}
}

void NeuralNetwork::Backpropagate(const Matrix& input, const Matrix& target)
{
	FeedForward(input, true);

	Layer* lastLayer = layers[numLayers - 1];
	// Compute cost derivative
	costFunction->Prime(*lastLayer->output, target, *costDerivative);
	// Feed partial delta of last layer
	costDerivative->CopyValuesTo(*lastLayer->delta);
	// Backpropagate
	for (int i = numLayers - 1; i > 0; --i)
		layers[i]->BackPropagate(*layers[i - 1]->output, layers[i - 1]->delta);
}

int NeuralNetwork::Predict(const Matrix& input)
{
	FeedForward(input, false);
	Matrix* output = layers[numLayers - 1]->output;
	int maxIndex = 0;
	for (int j = 0; j < output->matrixSize; ++j)
		if (output->data[j] > output->data[maxIndex])
			maxIndex = j;

	return maxIndex;
}

NeuralNetwork* NeuralNetwork::LoadFromFile(const QString& path)
{
	FILE* file = fopen(path.toStdString().c_str(), "rb");

	if (file == nullptr)
		throw std::runtime_error("Could not open file %s" + path.toStdString() + "\n");

	// Global parameters
	int numLayers;
	if (fread(&numLayers, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Could not read numLayers\n");
	float learningRate;
	if (fread(&learningRate, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Could not read learningRate\n");
	CostFunctions costFunction;
	if (fread(&costFunction, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Could not read costFunction\n");
	Optimizers optimizer;
	if (fread(&optimizer, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Could not read optimizer\n");

	// Create neural network
	NeuralNetwork* nn = new NeuralNetwork(optimizer, costFunction, learningRate);

	// Layers
	for (int i = 0; i < numLayers; ++i)
		nn->AddLayer(Layer::LoadStructureFromFile(file));

	nn->Compile();

	for (int i = 0; i < numLayers; ++i)
		nn->layers[i]->LoadParametersFromFile(file);

	fclose(file);

	return nn;
}

void NeuralNetwork::SaveToFile(const QString& path) const
{
	FILE* file = fopen(path.toStdString().c_str(), "wb");

	if (file == nullptr)
		throw std::runtime_error("Could not open file " + path.toStdString() + "\n");

	// Global parameters
	fwrite(&numLayers, sizeof(int), 1, file);
	// Learning rate
	fwrite(&learningRate, sizeof(float), 1, file);
	// Cost function
	fwrite(&costFunction->type, sizeof(int), 1, file);
	// Optimizer
	fwrite(&layers[1]->optimizer->type, sizeof(int), 1, file);

	// Layers
	for (int i = 0; i < numLayers; ++i)
		layers[i]->SaveStructureToFile(file);
	for (int i = 0; i < numLayers; ++i)
		layers[i]->SaveParametersToFile(file);

	fclose(file);
}

NeuralNetwork::NeuralNetwork(Optimizers optimizer, CostFunctions costFunction, const float learningRate) :
		numLayers(0),
		costFunction(CreateCostFunction(costFunction)),
		costDerivative(nullptr),
		learningRate(learningRate),
		optimizer(optimizer)
{
}

void NeuralNetwork::AddLayer(Layer* layer)
{
	numLayers++;
	layers.push_back(layer);
}

CostFunction* NeuralNetwork::CreateCostFunction(const CostFunctions type)
{
	switch (type)
	{
		case MSE:
			return (CostFunction*) new MSE_Cost();
		case CrossEntropy:
			return (CostFunction*) new CrossEntropyCost();
		default:
			throw std::runtime_error("Unknown cost function");
	}
}

void NeuralNetwork::Compile()
{
	LayerShape shape0(layers[0]->shape->dimensions[0], 1, 1);
	if (layers[1]->type & layers2DMask)
	{
		const float sqrt = std::sqrt((float) layers[0]->shape->dimensions[0]);
		const int rd = (int) sqrt;
		if ((float) rd != sqrt)
			throw std::runtime_error("Input layer size must be a perfect square as it is used for 2D layers");

		shape0.dimensions[1] = shape0.dimensions[2] = rd;
	}
	layers[0]->Compile(shape0, shape0, nullptr);

	for (int i = 1; i < numLayers; ++i)
	{
		Layer* l = layers[i];
		switch (l->type)
		{
			case FC:
			case Dropout:
			{
				LayerShape s(l->shape->dimensions[0], 1, 1);
				l->Compile(s, *layers[i - 1]->shape,
						   optimizer == Adam ? (Optimizer*) new AdamOptimizer() : (Optimizer*) new SGDOptimizer(
								   learningRate));
				break;
			}
			case Conv:
			case MaxPool:
			case Flatten:
				throw std::runtime_error("Not implemented");
			default:
				throw std::runtime_error("Unknown layer type");
		}
	}

	costDerivative = Matrix::CreateSameSize(*layers[numLayers - 1]->output);

	compiled = true;
}

void NeuralNetwork::UpdateAuxNetworks(BackpropagationThreadsArgs* args, const int numAuxThreads, const int batchOffset,
									  const int numInputsPerThread) const
{
	for (int i = 0; i < numAuxThreads; ++i)
		CopyValuesTo(*args[i].network);

	for (int i = 0; i < numAuxThreads; ++i)
		args[i].offset = batchOffset + i * numInputsPerThread;
}

void NeuralNetwork::BackPropagateInAuxThreads(pthread_t* threads, BackpropagationThreadsArgs* args,
											  const int numAuxThreads)
{
	for (int i = 0; i < numAuxThreads; ++i)
	{
		if (pthread_create(threads + i, nullptr, BackpropagationThread, args + i))
			throw std::runtime_error("pthread_create failed");
	}
}

void NeuralNetwork::BackPropagateInMainThread(const Matrix** trainingInputs, const Matrix** trainingOutputs,
											  const int offset,
											  const int numInputsInMainThread)
{
	for (int inputInd = 0; inputInd < numInputsInMainThread; inputInd++)
		Backpropagate(*trainingInputs[offset + inputInd], *trainingOutputs[offset + inputInd]);
}

void NeuralNetwork::WaitAuxThreads(pthread_t* threads, const int numAuxThreads)
{
	for (int i = 0; i < numAuxThreads; ++i)
	{
		if (pthread_join(threads[i], nullptr))
			throw std::runtime_error("pthread_join failed");
	}
}

void NeuralNetwork::UpdateParameters(const int numAuxThreads, BackpropagationThreadsArgs* threadsArgs,
									 const float batchScaleFactor)
{
	for (int i = 1; i < numLayers; ++i)
	{
		Layer* l = layers[i];

		// Add deltas from all threads
		for (int j = 0; j < numAuxThreads; ++j)
		{
			*l->deltaWeightsSum += *threadsArgs[j].network->layers[i]->deltaWeightsSum;
			*l->deltaBiasesSum += *threadsArgs[j].network->layers[i]->deltaBiasesSum;
		}

		*l->deltaBiasesSum *= batchScaleFactor;
		*l->deltaWeightsSum *= batchScaleFactor;

		l->optimizer->UpdateParameters(*l->weights, *l->deltaWeightsSum, 0);
		l->optimizer->UpdateParameters(*l->biases, *l->deltaBiasesSum, l->weights->matrixSize);

		l->deltaBiasesSum->Reset();
		l->deltaWeightsSum->Reset();
		/*MatReset(outputs[i]);
		MatReset(zs[i]);
		MatReset(delta[i]);
		MatReset(deltaWeights[i]);*/
	}
}
