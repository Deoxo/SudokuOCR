//
// Created by mat on 11/01/24.
//

#ifndef SUDOKUOCR_SOLVER_H
#define SUDOKUOCR_SOLVER_H


#include "Matrix.h"

namespace Solver
{
	Matrix* Solve(const Matrix& matrix);

	Matrix* ArrayToMatrix(const char* mat, int size);
	
	char* ParseInputFile(const char* path);
};


#endif //SUDOKUOCR_SOLVER_H
