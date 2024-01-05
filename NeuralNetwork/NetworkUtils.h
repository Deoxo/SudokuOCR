//
// Created by mat on 11/11/23.
//

#ifndef S3PROJECT_NETWORKUTILS_H
#define S3PROJECT_NETWORKUTILS_H

#include "../Matrix.h"

typedef struct LayerShape
{
	int* dimensions;

	LayerShape(int d1, int d2, int d3)
	{
		dimensions = new int[3];
		dimensions[0] = d1;
		dimensions[1] = d2;
		dimensions[2] = d3;
	}

	LayerShape(const LayerShape& other)
	{
		dimensions = new int[3];
		std::copy(other.dimensions, other.dimensions + 3, dimensions);
	}

	LayerShape& operator=(const LayerShape& other)
	{
		if (this == &other)
			return *this;
		std::copy(other.dimensions, other.dimensions + 3, dimensions);
		return *this;
	}

	~LayerShape()
	{
		delete[] dimensions;
	}
} LayerShape;

enum Optimizers
{
	NoOptimizer,
	SGD,
	Adam
};

enum CostFunctions
{
	MSE,
	CrossEntropy
};

enum ActivationFunctions
{
	Sigmoid,
	ReLU,
	Softmax,
	Tanh,
	LeakyReLU,
	None
};

enum LayerTypes
{
	Input = 0x0,
	FC = 0x1,
	Dropout = 0x2,
	Conv = 0x4,
	MaxPool = 0x8,
	Flatten = 0x16
};

const int layers2DMask = LayerTypes::Conv | LayerTypes::MaxPool;

float Sigmoid_(float x);

float SigmoidPrime_(float x);

float ReLU_(float x);

float ReLUPrime_(float x);

#endif //S3PROJECT_NETWORKUTILS_H
