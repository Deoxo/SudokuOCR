//
// Created by mat on 20/09/23.
//

#include <cmath>
#include "Layer.h"

/*Layer* CreateConvLayer(LayerShape* shape, ActivationFunction* activationFunction, Optimizer* optimizer,
                       LayerFunctions* functions)
{
    ConvLayer* convLayer = new ConvLayer();
    convLayer->base.shape = shape;
    convLayer->base.activationFunction = activationFunction;
    convLayer->base.functions = functions;
    convLayer->base.optimizer = optimizer;
    convLayer->base.Copy = CopyConvLayer;
    convLayer->base.CopyValuesTo = CopyConvValues;

    convLayer->numFilters = NUM_FILTERS;
    convLayer->filtersSize = FILTERS_SIZE;

    convLayer->base.weights = new Matrix(convLayer->filtersSize, convLayer->filtersSize, NUM_FILTERS);
    convLayer->base.biases = new Matrix(shape->dimensions[2], 1, 1);
    convLayer->base.output = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    convLayer->base.z = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    convLayer->base.delta = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    convLayer->base.deltaBiases = new Matrix(shape->dimensions[2], 1, 1);
    convLayer->base.deltaWeights = new Matrix(convLayer->filtersSize, convLayer->filtersSize, NUM_FILTERS);
    convLayer->base.activationPrime = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    convLayer->base.deltaWeightsSum = new Matrix(convLayer->filtersSize, convLayer->filtersSize, NUM_FILTERS);
    convLayer->base.deltaBiasesSum = new Matrix(shape->dimensions[2], 1, 1);

    // Initialize weights and biases
    for (int j = 0; j < convLayer->base.weights->matrixSize; ++j)
        convLayer->base.weights->data[j] = ((double) rand()) / ((double) RAND_MAX) * 2 - 1;
    for (int j = 0; j < convLayer->base.biases->matrixSize; ++j)
        convLayer->base.biases->data[j] = ((double) rand()) / ((double) RAND_MAX) * 2 - 1;

    // Make sure the deltaSums are initialized to 0
    convLayer->base.deltaWeightsSum->Reset();
    convLayer->base.deltaBiasesSum->Reset();

    return (Layer*) convLayer;
}

void CopyMaxPoolValues(const Layer* layer, Layer* destination)
{
    MaxPoolLayer* original = (MaxPoolLayer*) layer;
    MaxPoolLayer* dest = (MaxPoolLayer*) destination;
    original->base.output->CopyValuesTo(*dest->base.output);
    original->base.delta->CopyValuesTo(*dest->base.delta);
}

Layer* CreateMaxPoolingLayer(LayerShape* shape, ActivationFunction* activationFunction, Optimizer* optimizer,
                             LayerFunctions* functions, const int stride, const int filterSize)
{
    MaxPoolLayer* maxPoolLayer = new MaxPoolLayer();
    maxPoolLayer->base.shape = shape;
    maxPoolLayer->base.activationFunction = activationFunction;
    maxPoolLayer->base.functions = functions;
    maxPoolLayer->base.optimizer = optimizer;
    maxPoolLayer->base.Copy = CopyMaxPoolLayer;
    maxPoolLayer->base.CopyValuesTo = CopyMaxPoolValues;

    maxPoolLayer->stride = stride;
    maxPoolLayer->filterSize = filterSize;

    maxPoolLayer->base.weights = nullptr;
    maxPoolLayer->base.biases = nullptr;
    maxPoolLayer->base.output = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    maxPoolLayer->base.z = nullptr;
    maxPoolLayer->base.delta = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    maxPoolLayer->base.deltaBiases = nullptr;
    maxPoolLayer->base.deltaWeights = nullptr;
    maxPoolLayer->base.activationPrime = nullptr;
    maxPoolLayer->base.deltaWeightsSum = nullptr;
    maxPoolLayer->base.deltaBiasesSum = nullptr;

    return (Layer*) maxPoolLayer;
}

Layer* CreateFlattenLayer(LayerShape* shape, LayerFunctions* functions)
{
    FlattenLayer* layer = new FlattenLayer();
    layer->base.shape = shape;
    layer->base.functions = functions;
    layer->base.activationFunction = nullptr;
    layer->base.optimizer = nullptr;

    layer->base.weights = nullptr;
    layer->base.biases = nullptr;
    layer->base.output = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    layer->base.z = nullptr;
    layer->base.delta = new Matrix(shape->dimensions[0], shape->dimensions[1], shape->dimensions[2]);
    layer->base.deltaBiases = nullptr;
    layer->base.deltaWeights = nullptr;
    layer->base.activationPrime = nullptr;
    layer->base.deltaWeightsSum = nullptr;
    layer->base.deltaBiasesSum = nullptr;

    return (Layer*) layer;
}

void CopyConvValues(const Layer* layer, Layer* destination)
{
    ConvLayer* original = (ConvLayer*) layer;
    ConvLayer* dest = (ConvLayer*) destination;
    original->base.weights->CopyValuesTo(*dest->base.weights);
    original->base.biases->CopyValuesTo(*dest->base.biases);
    original->base.output->CopyValuesTo(*dest->base.output);
    original->base.z->CopyValuesTo(*dest->base.z);
    original->base.delta->CopyValuesTo(*dest->base.delta);
    original->base.deltaBiases->CopyValuesTo(*dest->base.deltaBiases);
    original->base.deltaWeights->CopyValuesTo(*dest->base.deltaWeights);
    original->base.activationPrime->CopyValuesTo(*dest->base.activationPrime);
    original->base.deltaWeightsSum->CopyValuesTo(*dest->base.deltaWeightsSum);
    original->base.deltaBiasesSum->CopyValuesTo(*dest->base.deltaBiasesSum);
}

Layer* CopyConvLayer(const Layer* layer)
{
    ConvLayer* convLayer = (ConvLayer*) layer;
    ConvLayer* copy = new ConvLayer();

    copy->base.functions = CopyLayerFunctions(layer->functions);
    copy->base.shape = CopyLayerShape(layer->shape);
    copy->base.activationFunction = CopyActivationFunction(layer->activationFunction);
    copy->base.optimizer = layer->optimizer->Copy(layer->optimizer);

    copy->filtersSize = convLayer->filtersSize;
    copy->numFilters = convLayer->numFilters;

    copy->base.weights = Matrix::CreateSameSize(*convLayer->base.weights);
    copy->base.biases = Matrix::CreateSameSize(*convLayer->base.biases);
    copy->base.output = Matrix::CreateSameSize(*convLayer->base.output);
    copy->base.z = Matrix::CreateSameSize(*convLayer->base.z);
    copy->base.delta = Matrix::CreateSameSize(*convLayer->base.delta);
    copy->base.deltaBiases = Matrix::CreateSameSize(*convLayer->base.deltaBiases);
    copy->base.deltaWeights = Matrix::CreateSameSize(*convLayer->base.deltaWeights);
    copy->base.activationPrime = Matrix::CreateSameSize(*convLayer->base.activationPrime);
    copy->base.deltaWeightsSum = Matrix::CreateSameSize(*convLayer->base.deltaWeightsSum);
    copy->base.deltaBiasesSum = Matrix::CreateSameSize(*convLayer->base.deltaBiasesSum);

    return (Layer*) copy;
}

Layer* CopyMaxPoolLayer(const Layer* layer)
{
    MaxPoolLayer* maxPoolLayer = (MaxPoolLayer*) layer;
    MaxPoolLayer* copy = new MaxPoolLayer();

    copy->base.functions = CopyLayerFunctions(layer->functions);
    copy->base.shape = CopyLayerShape(layer->shape);
    copy->base.activationFunction = CopyActivationFunction(layer->activationFunction);
    copy->base.optimizer = layer->optimizer->Copy(layer->optimizer);

    copy->filterSize = maxPoolLayer->filterSize;
    copy->stride = maxPoolLayer->stride;

    copy->base.weights = nullptr;
    copy->base.biases = nullptr;
    copy->base.output = Matrix::CreateSameSize(*maxPoolLayer->base.output);
    copy->base.z = nullptr;
    copy->base.delta = Matrix::CreateSameSize(*maxPoolLayer->base.delta);
    copy->base.deltaBiases = nullptr;
    copy->base.deltaWeights = nullptr;
    copy->base.activationPrime = nullptr;
    copy->base.deltaWeightsSum = nullptr;
    copy->base.deltaBiasesSum = nullptr;

    return (Layer*) copy;
}

Layer* CopyFlattenLayer(const Layer* layer)
{
    FlattenLayer* flattenLayer = (FlattenLayer*) layer;
    FlattenLayer* copy = new FlattenLayer();

    copy->base.functions = CopyLayerFunctions(layer->functions);
    copy->base.shape = CopyLayerShape(layer->shape);
    copy->base.activationFunction = nullptr;
    copy->base.optimizer = nullptr;

    copy->base.weights = nullptr;
    copy->base.biases = nullptr;
    copy->base.output = Matrix::CreateSameSize(*flattenLayer->base.output);
    copy->base.z = nullptr;
    copy->base.delta = Matrix::CreateSameSize(*flattenLayer->base.delta);
    copy->base.deltaBiases = nullptr;
    copy->base.deltaWeights = nullptr;
    copy->base.activationPrime = nullptr;
    copy->base.deltaWeightsSum = nullptr;
    copy->base.deltaBiasesSum = nullptr;

    return (Layer*) copy;
}

void ConvFeedForward(Layer* layer, const Matrix* input)
{
    ConvLayer* convLayer = (ConvLayer*) layer;

    for (int i = 0; i < input->dims; ++i)
    {
        for (int j = 0; j < convLayer->numFilters; ++j)
        {
            Matrix::ValidConvolution(*input, *convLayer->base.weights, *layer->z);
            *layer->z += layer->biases->data[i];
            layer->activationFunction->function(layer->z, layer->output);

            convLayer->base.weights->GoToNextMatrix();
            layer->z->GoToNextMatrix();
            layer->output->GoToNextMatrix();
        }
    }
    convLayer->base.weights->ResetOffset();
    layer->z->ResetOffset();
    layer->output->ResetOffset();
}

void ConvBackPropagate(Layer* layer, const Matrix* pastOutput, Matrix* nextDelta)
{
    ConvLayer* convLayer = (ConvLayer*) layer;
    for (int i = 0; i < convLayer->numFilters; ++i)
    {
        layer->activationFunction->derivative(layer->z, layer->activationPrime); // Compute delta
        Matrix::LinearMultiply(*layer->activationPrime, *layer->delta, *layer->delta); // Uses previously stored values
    }

    for (int i = 0; i < convLayer->numFilters; ++i)
    {
        Matrix::ValidConvolution(*pastOutput, *layer->delta, *layer->deltaWeights); // Compute delta weights
        *layer->deltaWeightsSum += *layer->deltaWeights; // Sum delta weights

        Matrix::TransposeFullConvolution(*convLayer->base.weights, *layer->delta,
                                         *nextDelta); // Compute partial delta of next layer

        const float db = layer->delta->Sum(); // Compute delta biases
        convLayer->base.deltaBiases->data[i] = db;
        convLayer->base.deltaBiasesSum->data[i] += db; // Sum delta biases

        layer->delta->GoToNextMatrix();
        layer->deltaWeights->GoToNextMatrix();
        layer->deltaWeightsSum->GoToNextMatrix();
    }

    layer->delta->ResetOffset();
    layer->deltaWeights->ResetOffset();
    layer->deltaWeightsSum->ResetOffset();
}

void MaxPoolFeedForward(Layer* layer, const Matrix* input)
{
    MaxPoolLayer* maxPoolLayer = (MaxPoolLayer*) layer;
    Matrix::MaxPool(*input, maxPoolLayer->filterSize, maxPoolLayer->stride, *maxPoolLayer->base.output);
}

void MaxPoolBackPropagate(Layer* layer, const Matrix* pastOutput, Matrix* nextDelta)
{
    MaxPoolLayer* maxPoolLayer = (MaxPoolLayer*) layer;
    const int pr = pastOutput->rows;
    const int pc = pastOutput->cols;
    for (int m = 0; m < pastOutput->dims; m++)
    {
        for (int i = 0; i < layer->shape->dimensions[0]; ++i)
        {
            for (int j = 0; j < layer->shape->dimensions[1]; ++j)
            {
                for (int k = 0; k < maxPoolLayer->filterSize; ++k)
                {
                    for (int l = 0; l < maxPoolLayer->filterSize; ++l)
                    {
                        const int r = i * maxPoolLayer->stride + k;
                        if (r >= pr)
                            continue;
                        const int c = j * maxPoolLayer->stride + l;
                        if (c >= pc)
                            continue;

                        if ((*pastOutput)(r, c) == (*layer->output)(i, j))
                            (*nextDelta)(r, c) = (*layer->delta)(i, j);
                        else
                            (*nextDelta)(r, c) = 0;
                    }
                }
            }
        }
        ((Matrix*) pastOutput)->GoToNextMatrix();
        layer->output->GoToNextMatrix();
        nextDelta->GoToNextMatrix();
        layer->delta->GoToNextMatrix();
    }

    ((Matrix*) pastOutput)->ResetOffset();
    layer->output->ResetOffset();
    nextDelta->ResetOffset();
    layer->delta->ResetOffset();
}

void FlattenFeedForward(Layer* layer, const Matrix* input)
{
    Matrix* o = layer->output;
    o->rows = layer->shape->dimensions[0];
    o->cols = layer->shape->dimensions[1];
    o->dims = layer->shape->dimensions[2];
    o->matrixSize = o->rows * o->cols;

    Matrix* d = layer->delta;
    d->rows = layer->shape->dimensions[0];
    d->cols = layer->shape->dimensions[1];
    d->dims = layer->shape->dimensions[2];
    d->matrixSize = d->rows * d->cols;

    (void) input;
}

void FlattenBackPropagate(Layer* layer, const Matrix* pastOutput, Matrix* nextDelta)
{
    Matrix* o = layer->output;
    o->rows = pastOutput->rows;
    o->cols = pastOutput->cols;
    o->dims = pastOutput->dims;
    o->matrixSize = pastOutput->matrixSize;

    Matrix* d = layer->delta;
    d->rows = pastOutput->rows;
    d->cols = pastOutput->cols;
    d->dims = pastOutput->dims;
    d->matrixSize = pastOutput->matrixSize;

    (void) nextDelta;
}*/

Layer::Layer(LayerTypes type) :
		type(type),
		optimizer(nullptr),
		shape(new LayerShape(0, 0, 0))
{}

Layer::~Layer()
{
	delete output;
	delete z;
	delete delta;
	delete deltaBiases;
	delete deltaWeights;
	delete activationPrime;
	delete deltaWeightsSum;
	delete deltaBiasesSum;
	delete weights;
	delete biases;

	delete optimizer;
	delete shape;
}

ActivationFunction* Layer::CreateActivationFunction(const ActivationFunctions type)
{
	switch (type)
	{
		case None:
			return new NoActivation();
		case Sigmoid:
			return new SigmoidActivation();
		case ReLU:
			return new ReLUActivation();
		case Softmax:
			return new SoftmaxActivation();
		case Tanh:
			return new TanhActivation();
		case LeakyReLU:
			return new LeakyReLUActivation();

		default:
			throw std::runtime_error("Invalid activation function type");
	}
}

void Layer::Compile(const LayerShape& shape_, const LayerShape& previousLayerShape [[maybe_unused]],
					Optimizer* optimizer_)
{
	*this->shape = shape_;
	optimizer = optimizer_;
}

Layer* Layer::LoadStructureFromFile(FILE* file)
{
	// Layer type
	LayerTypes layerType;
	if (fread(&layerType, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Could not read layer type\n");

	// Create layer
	Layer* l;
	switch (layerType)
	{
		case Input:
			l = new InputLayer(file);
			break;
		case FC:
			l = new FCL(file);
			break;
		case Dropout:
			l = new DropoutLayer(file);
			break;
		case Conv:
		case MaxPool:
		case Flatten:
			throw std::runtime_error("Not implemented");
		default:
			throw std::runtime_error("Unknown layer type");
	}

	return l;
}

void Layer::SaveStructureToFile(FILE* file)
{
	fwrite(&type, sizeof(int), 1, file);
}

void Layer::SaveParametersToFile(FILE* file)
{
	if (optimizer != nullptr)
		optimizer->SaveParametersToFile(file);
}

void Layer::LoadParametersFromFile(FILE* file)
{
	if (optimizer != nullptr)
		optimizer->LoadParametersFromFile(file);
}

void InputLayer::FeedForward(const Matrix& input, const bool train [[maybe_unused]])
{
	input.CopyValuesTo(*output);
}


InputLayer::InputLayer(int numNeurons) : Layer(Input)
{
	shape->dimensions[0] = numNeurons;
}

InputLayer::InputLayer(FILE* file) : Layer(Input)
{
	if (fread(&shape->dimensions[0], sizeof(int), 1, file) != 1) // Load number of neurons
		throw std::runtime_error("Could not read number of neurons\n");
}

void
InputLayer::Compile(const LayerShape& shape, const LayerShape& previousLayerShape, Optimizer* optimizer)
{
	Layer::Compile(shape, previousLayerShape, optimizer);
	output = new Matrix(shape.dimensions[0], shape.dimensions[1], shape.dimensions[2]);
}

void
InputLayer::BackPropagate(const Matrix& pastOutput [[maybe_unused]], Matrix* nextDelta [[maybe_unused]])
{

}

Layer* InputLayer::Copy() const
{
	return new InputLayer(shape->dimensions[0]);
}

void InputLayer::CopyValuesTo(const Layer& other [[maybe_unused]])
{

}

void InputLayer::SaveStructureToFile(FILE* file)
{
	Layer::SaveStructureToFile(file);
	fwrite(&shape->dimensions[0], sizeof(int), 1, file); // Save number of neurons
}

FCL::FCL(ActivationFunctions activationFunction, const int numNeurons) : Layer(FC),
																		 activationFunction(CreateActivationFunction(
																				 activationFunction))
{
	shape->dimensions[0] = numNeurons;
}

FCL::FCL(FILE* file) : Layer(FC)
{
	ActivationFunctions activationFunctionType;
	if (fread(&activationFunctionType, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Could not read activation function type\n");
	if (fread(&shape->dimensions[0], sizeof(int), 1, file) != 1) // Load number of neurons
		throw std::runtime_error("Could not read number of neurons\n");
	activationFunction = CreateActivationFunction(activationFunctionType);
}

void
FCL::Compile(const LayerShape& shape, const LayerShape& previousLayerShape, Optimizer* optimizer)
{
	Layer::Compile(shape, previousLayerShape, optimizer);
	const int prevLayerNumNeurons = previousLayerShape.dimensions[0] * previousLayerShape.dimensions[1] *
									previousLayerShape.dimensions[2];


	// Allocate the matrices
	weights = new Matrix(shape.dimensions[0], prevLayerNumNeurons, 1);
	biases = new Matrix(shape.dimensions[0], 1, 1);
	output = new Matrix(shape.dimensions[0], 1, 1);
	z = new Matrix(shape.dimensions[0], 1, 1);
	delta = new Matrix(shape.dimensions[0], 1, 1);
	deltaBiases = new Matrix(shape.dimensions[0], 1, 1);
	deltaWeights = new Matrix(shape.dimensions[0], prevLayerNumNeurons, 1);
	activationPrime = new Matrix(shape.dimensions[0], 1, 1);
	deltaWeightsSum = new Matrix(shape.dimensions[0], prevLayerNumNeurons, 1);
	deltaBiasesSum = new Matrix(shape.dimensions[0], 1, 1);

	// Initialize weights and biases
	for (int j = 0; j < weights->matrixSize; ++j)
		weights->data[j] = ((double) rand()) / ((double) RAND_MAX) * 2 - 1;
	for (int j = 0; j < biases->matrixSize; ++j)
		biases->data[j] = ((double) rand()) / ((double) RAND_MAX) * 2 - 1;

	// Make sure the deltaSums are initialized to 0
	deltaWeightsSum->Reset();
	deltaBiasesSum->Reset();

	const int optNumsParams = weights->size + biases->size;
	optimizer->Compile(optNumsParams);
}

void FCL::FeedForward(const Matrix& input, const bool train [[maybe_unused]])
{
	Matrix::Multiply(*weights, input, *z);
	*z += *biases;
	activationFunction->Function(*z, *output);
}

void FCL::BackPropagate(const Matrix& pastOutput, Matrix* nextDelta)
{
	activationFunction->Prime(*z, *activationPrime); // Compute delta
	Matrix::LinearMultiply(*activationPrime, *delta, *delta); // Uses previously stored values
	delta->CopyValuesTo(*deltaBiases);
	*deltaBiasesSum += *deltaBiases; // Sum delta biases
	Matrix::MultiplyByTranspose(*delta, pastOutput, *deltaWeights); // Compute delta weights
	*deltaWeightsSum += *deltaWeights; // Sum delta weights

	// Compute partial delta of next layer
	if (nextDelta != nullptr) // If next layer is the input layer, nextDelta is nullptr
		Matrix::TransposeMultiply(*weights, *delta, *nextDelta);

}

Layer* FCL::Copy() const
{
	return new FCL(activationFunction->type, shape->dimensions[0]);
}

void FCL::CopyValuesTo(const Layer& other)
{
	weights->CopyValuesTo(*other.weights);
	biases->CopyValuesTo(*other.biases);
	output->CopyValuesTo(*other.output);
	z->CopyValuesTo(*other.z);
	delta->CopyValuesTo(*other.delta);
	deltaBiases->CopyValuesTo(*other.deltaBiases);
	deltaWeights->CopyValuesTo(*other.deltaWeights);
	activationPrime->CopyValuesTo(*other.activationPrime);
	deltaWeightsSum->CopyValuesTo(*other.deltaWeightsSum);
	deltaBiasesSum->CopyValuesTo(*other.deltaBiasesSum);
}

void FCL::SaveStructureToFile(FILE* file)
{
	Layer::SaveStructureToFile(file);
	fwrite(&activationFunction->type, sizeof(int), 1, file);
	fwrite(&shape->dimensions[0], sizeof(int), 1, file); // Save number of neurons
}

void FCL::LoadParametersFromFile(FILE* file)
{
	Layer::LoadParametersFromFile(file);
	weights->LoadFromFile(file);
	biases->LoadFromFile(file);
}

void FCL::SaveParametersToFile(FILE* file)
{
	Layer::SaveParametersToFile(file);
	weights->SaveToFile(file);
	biases->SaveToFile(file);
}

DropoutLayer::DropoutLayer(ActivationFunctions activationFunction, const int numNeurons, const float dropoutRate) :
		FCL(activationFunction, numNeurons), dropoutRate(dropoutRate)
{
	type = Dropout;
}

DropoutLayer::DropoutLayer(FILE* file) : FCL(file), dropoutRate(0)
{
	if (fread(&dropoutRate, sizeof(float), 1, file) != 1)
		throw std::runtime_error("Could not read dropout rate\n");
}

void DropoutLayer::FeedForward(const Matrix& input, const bool training)
{
	FCL::FeedForward(input, training);
	if (training)
	{
		for (int i = 0; i < output->size; ++i)
			if (((double) rand()) / ((double) RAND_MAX) < dropoutRate)
				output->data[i] = 0;
	}
}

Layer* DropoutLayer::Copy() const
{
	return new DropoutLayer(activationFunction->type, shape->dimensions[0], dropoutRate);
}

void DropoutLayer::SaveStructureToFile(FILE* file)
{
	FCL::SaveStructureToFile(file);
	fwrite(&dropoutRate, sizeof(float), 1, file);
}