//
// Created by mat on 20/09/23.
//

#ifndef S3PROJECT_LAYER_H
#define S3PROJECT_LAYER_H

#include "NetworkUtils.h"
#include "ActivationFunction.h"
#include "Optimizer.h"

class Layer
{
protected:
    explicit Layer(LayerTypes type);

    [[nodiscard]] static ActivationFunction* CreateActivationFunction(ActivationFunctions type);

public:
    LayerTypes type;

    Matrix* output = nullptr;
    Matrix* z = nullptr;
    Matrix* delta = nullptr;
    Matrix* deltaBiases = nullptr;
    Matrix* deltaWeights = nullptr;
    Matrix* activationPrime = nullptr;

    Matrix* deltaWeightsSum = nullptr;
    Matrix* deltaBiasesSum = nullptr;

    Matrix* weights = nullptr;
    Matrix* biases = nullptr;

    LayerShape* shape;
    Optimizer* optimizer;


    [[nodiscard]] static Layer* LoadStructureFromFile(FILE* file);

    virtual void LoadParametersFromFile(FILE* file);

    virtual void SaveStructureToFile(FILE* file);

    virtual void SaveParametersToFile(FILE* file);

    virtual void
    Compile(const LayerShape& shape_, const LayerShape& previousLayerShape, Optimizer* optimizer_);

    virtual ~Layer();

    [[nodiscard]] virtual Layer* Copy() const = 0;

    virtual void CopyValuesTo(const Layer& other) = 0;

    virtual void FeedForward(const Matrix& input, bool train) = 0;

    // This function assumes that w[l+1] * delta[l+1] (or cost prime for last layer) is stored into layer->delta
    virtual void BackPropagate(const Matrix& pastOutput, Matrix* nextDelta) = 0;
};

class InputLayer : public Layer
{
public:
    explicit InputLayer(int numNeurons);

    explicit InputLayer(FILE* file);

    void Compile(const LayerShape& shape, const LayerShape& previousLayerShape, Optimizer* optimizer) override;

    void FeedForward(const Matrix& input, bool train) override;

    void BackPropagate(const Matrix& pastOutput, Matrix* nextDelta) override;

    [[nodiscard]] Layer* Copy() const override;

    void CopyValuesTo(const Layer& other) override;

    void SaveStructureToFile(FILE* file) override;
};

class FCL : public Layer
{
public:
    explicit FCL(ActivationFunctions activationFunction, int numNeurons);

    explicit FCL(FILE* file);

    [[nodiscard]] Layer* Copy() const override;

    void CopyValuesTo(const Layer& other) override;

    void Compile(const LayerShape& shape, const LayerShape& previousLayerShape, Optimizer* optimizer) override;

    void FeedForward(const Matrix& input, bool train) override;

    void BackPropagate(const Matrix& pastOutput, Matrix* nextDelta) override;

    void SaveStructureToFile(FILE* file) override;

    void LoadParametersFromFile(FILE* file) override;

    void SaveParametersToFile(FILE* file) override;

    ActivationFunction* activationFunction;
};

class DropoutLayer : public FCL
{
public:
    DropoutLayer(ActivationFunctions activationFunction, int numNeurons, float dropoutRate);

    explicit DropoutLayer(FILE* file);

    void FeedForward(const Matrix& input, bool training) override;

    void SaveStructureToFile(FILE* file) override;

    float dropoutRate;
private:
    [[nodiscard]] Layer* Copy() const override;
};

/*class ConvLayer : Layer
{
public:
    ConvLayer(const LayerShape& shape, const LayerShape& previousLayerShape, Optimizer* optimizer,
              ActivationFunction* activationFunction);

    int numFilters, filtersSize;

    //int stride;
    //int padding;
};

class FlattenLayer : Layer
{

    FlattenLayer(const LayerShape& shape, const LayerShape& previousLayerShape, Optimizer* optimizer,
                 ActivationFunction* activationFunction);
};

class MaxPoolLayer : Layer
{
public:
    MaxPoolLayer(const LayerShape& shape, const LayerShape& previousLayerShape, Optimizer* optimizer,
                 ActivationFunction* activationFunction, int stride, int filterSize);

    int stride, filterSize;
};*/

/*Layer* CreateConvLayer(LayerShape* shape, ActivationFunction* activationFunction, Optimizer* optimizer,
                       LayerFunctions* functions);

Layer* CreateMaxPoolingLayer(LayerShape* shape, ActivationFunction* activationFunction, Optimizer* optimizer,
                             LayerFunctions* functions, int stride, int filterSize);

Layer* CreateFlattenLayer(LayerShape* shape, LayerFunctions* functions);

void CopyConvValues(const Layer* layer, Layer* destination);

void CopyMaxPoolValues(const Layer* layer, Layer* destination);

Layer* CopyConvLayer(const Layer* layer);

Layer* CopyMaxPoolLayer(const Layer* layer);

Layer* CopyFlattenLayer(const Layer* layer);

void ConvFeedForward(Layer* layer, const Matrix* input);

void ConvBackPropagate(Layer* layer, const Matrix* pastOutput, Matrix* nextDelta);

void MaxPoolFeedForward(Layer* layer, const Matrix* input);

void MaxPoolBackPropagate(Layer* layer, const Matrix* pastOutput, Matrix* nextDelta);

void FlattenFeedForward(Layer* layer, const Matrix* input);

void FlattenBackPropagate(Layer* layer, const Matrix* pastOutput, Matrix* nextDelta);*/

#endif //S3PROJECT_LAYER_H
