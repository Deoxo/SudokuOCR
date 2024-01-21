//
// Created by mat on 18/01/24.
//

#ifndef SUDOKUOCR_MATH_H
#define SUDOKUOCR_MATH_H

#include <QPoint>
#include "Matrix.h"

namespace Math
{
	QPoint RotateQPoint(const QPoint& point, const QPoint& center, float angleDegrees);

	Matrix* Get3x3MatrixInverse(const Matrix& m);

	Matrix* GaussianElimination(const Matrix& a, const Matrix& b, int n);
}


#endif //SUDOKUOCR_MATH_H
