//
// Created by mat on 31/12/23.
//

#include "CostFunction.h"

CostFunction::CostFunction(const CostFunctions& type) : type(type)
{

}

MSE_Cost::MSE_Cost() : CostFunction(MSE)
{

}

float MSE_Cost::Function(const Matrix& m, const Matrix& target)
{
    float sum = 0;
    for (int i = 0; i < m.matrixSize; ++i)
    {
        const float diff = m.data[i] - target.data[i];
        sum += diff * diff;
    }
    return sum;
}

void MSE_Cost::Prime(const Matrix& m, const Matrix& target, Matrix& output)
{
    for (int i = 0; i < m.matrixSize; ++i)
        output.data[i] = 2 * (m.data[i] - target.data[i]);
}

CostFunction* MSE_Cost::Copy() const
{
    return new MSE_Cost();
}

CrossEntropyCost::CrossEntropyCost() : CostFunction(CrossEntropy)
{

}

float CrossEntropyCost::Function(const Matrix& m, const Matrix& target)
{
    float sum = 0;

    for (int i = 0; i < m.matrixSize; ++i)
        sum += target.data[i] * logf(m.data[i] + 1e-15f) + (1 - target.data[i]) * logf(1 - m.data[i] + 1e-15f);

    return -sum;
}

void CrossEntropyCost::Prime(const Matrix& m, const Matrix& target, Matrix& output)
{
    for (int i = 0; i < output.matrixSize; ++i)
        output.data[i] = target.data[i] == 1 ? -1 + m.data[i] : m.data[i];
}

CostFunction* CrossEntropyCost::Copy() const
{
    return new CrossEntropyCost();
}
