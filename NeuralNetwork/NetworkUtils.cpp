//
// Created by mat on 11/11/23.
//

#include "NetworkUtils.h"
#include "Layer.h"
#include <mm_malloc.h>

float Sigmoid_(const float x)
{
	return 1.f / (1.f + expf(-x));
}

float SigmoidPrime_(const float x)
{
	const float sx = Sigmoid_(x);
	return sx * (1.f - sx);
}

float ReLU_(const float x)
{
	return x > 0 ? x : 0;
}

float ReLUPrime_(const float x)
{
	return x > 0 ? 1 : 0;
}
