//
// Created by mat on 31/12/23.
//

#include "ActivationFunction.h"
#include "NetworkUtils.h"

ActivationFunction::ActivationFunction(const ActivationFunctions& type) : type(type)
{

}


void SigmoidActivation::Function(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = Sigmoid_(input.data[i]);
}

void SigmoidActivation::Prime(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = SigmoidPrime_(input.data[i]);
}

ActivationFunction* SigmoidActivation::Copy() const
{
    return new SigmoidActivation();
}

void ReLUActivation::Function(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = ReLU_(input.data[i]);
}

void ReLUActivation::Prime(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = ReLUPrime_(input.data[i]);
}

ActivationFunction* ReLUActivation::Copy() const
{
    return new ReLUActivation();
}

void SoftmaxActivation::Function(const Matrix& input, Matrix& output)
{
    float sum = 0;
    for (int i = 0; i < input.matrixSize; ++i)
        sum += expf(input.data[i]);
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = expf(input.data[i]) / sum;
}

void SoftmaxActivation::Prime(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = 1;
}

ActivationFunction* SoftmaxActivation::Copy() const
{
    return new SoftmaxActivation();
}

void TanhActivation::Function(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = tanhf(input.data[i]);
}

void TanhActivation::Prime(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
    {
        const float t = tanhf(input.data[i]);
        output.data[i] = 1 - t * t;
    }
}

ActivationFunction* TanhActivation::Copy() const
{
    return new TanhActivation();
}

void LeakyReLUActivation::Function(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = input.data[i] > 0 ? input.data[i] : 0.01f * input.data[i];
}

void LeakyReLUActivation::Prime(const Matrix& input, Matrix& output)
{
    for (int i = 0; i < input.matrixSize; ++i)
        output.data[i] = input.data[i] > 0 ? 1 : 0.01f;
}

ActivationFunction* LeakyReLUActivation::Copy() const
{
    return new LeakyReLUActivation();
}

NoActivation::NoActivation() : ActivationFunction(None)
{

}

ActivationFunction* NoActivation::Copy() const
{
    return new NoActivation();
}

void NoActivation::Function(const Matrix& input [[maybe_unused]], Matrix& output [[maybe_unused]])
{

}

void NoActivation::Prime(const Matrix& input [[maybe_unused]], Matrix& output [[maybe_unused]])
{

}
