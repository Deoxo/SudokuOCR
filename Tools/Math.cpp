//
// Created by mat on 18/01/24.
//

#include <QtMath>
#include "Math.h"

namespace Math
{
	// Solve a system of linear equations using Gaussian Elimination with Partial Pivoting
	Matrix* GaussianElimination(const Matrix& a, const Matrix& b, const int n)
	{
		// Check if the matrices are compatible
		if (a.rows != n || a.cols != n || b.rows != n || b.cols != 1)
			throw std::runtime_error("Invalid input");

		Matrix* res = new Matrix(n, 1);

		int i, j, k;

		// Forward Elimination with Partial Pivoting
		for (i = 0; i < n; i++)
		{
			// Partial Pivoting: Find the maximum element in the pivot column
			int pivotRow = i;
			for (j = i + 1; j < n; j++)
			{
				if (std::abs(a.data[j * n + i]) > std::abs(a.data[pivotRow * n + i]))
					pivotRow = j;
			}

			// Swap rows if needed
			if (pivotRow != i)
			{
				for (k = 0; k < n; k++)
					std::swap(a.data[i * n + k], a.data[pivotRow * n + k]);

				std::swap(b.data[i], b.data[pivotRow]);
			}

			// Check for a zero pivot (avoid division by zero)
			if (a.data[i * n + i] == 0.0)
			{
				// Handle zero pivot (potentially return an error code)
				delete res;
				throw std::runtime_error("Invalid input");
			}

			// Perform forward elimination with the non-zero pivot
			for (j = i + 1; j < n; j++)
			{
				const float ratio = a.data[j * n + i] / a.data[i * n + i];
				for (k = 0; k < n; k++)
					a.data[j * n + k] -= ratio * a.data[i * n + k];

				b.data[j] -= ratio * b.data[i];
			}
		}

		// Back Substitution
		for (i = n - 1; i >= 0; i--)
		{
			res->data[i] = b.data[i];
			for (j = i + 1; j < n; j++)
				res->data[i] -= a.data[i * n + j] * res->data[j];

			res->data[i] /= a.data[i * n + i];
		}

		return res;
	}

	// Compute the inverse of a 3x3 matrix by dividing the adjoint by the determinant
	Matrix* Get3x3MatrixInverse(const Matrix& m)
	{
		const float det = m.data[0] * (m.data[4] * m.data[8] - m.data[5] * m.data[7]) -
						  m.data[1] * (m.data[3] * m.data[8] - m.data[5] * m.data[6]) +
						  m.data[2] * (m.data[3] * m.data[7] - m.data[4] * m.data[6]);

		Matrix* inv = new Matrix(3, 3);
		inv->data[0] = m.data[4] * m.data[8] - m.data[5] * m.data[7];
		inv->data[1] = m.data[2] * m.data[7] - m.data[1] * m.data[8];
		inv->data[2] = m.data[1] * m.data[5] - m.data[2] * m.data[4];
		inv->data[3] = m.data[5] * m.data[6] - m.data[3] * m.data[8];
		inv->data[4] = m.data[0] * m.data[8] - m.data[2] * m.data[6];
		inv->data[5] = m.data[2] * m.data[3] - m.data[0] * m.data[5];
		inv->data[6] = m.data[3] * m.data[7] - m.data[4] * m.data[6];
		inv->data[7] = m.data[1] * m.data[6] - m.data[0] * m.data[7];
		inv->data[8] = m.data[0] * m.data[4] - m.data[1] * m.data[3];

		*inv /= det;

		return inv;
	}

	QPoint RotateQPoint(const QPoint& point, const QPoint& center, const float angleDegrees)
	{
		const qreal angleRadians = qDegreesToRadians(angleDegrees);
		const int rotatedX = qRound(center.x() + (point.x() - center.x()) * qCos(angleRadians) -
									(point.y() - center.y()) * qSin(angleRadians));
		const int rotatedY = qRound(center.y() + (point.x() - center.x()) * qSin(angleRadians) +
									(point.y() - center.y()) * qCos(angleRadians));

		return {rotatedX, rotatedY};
	}
}