//
// Created by mat on 27/12/23.
//

#ifndef SUDOKUOCR_IMAGERY_H
#define SUDOKUOCR_IMAGERY_H

#include <QString>
#include <QPointF>
#include "Tools/Matrix.h"
#include "../Tools/List.h"

typedef struct HoughLine
{
	float theta;
	float rho;
} HoughLine;

typedef struct Line
{
	int x1;
	int y1;
	int x2;
	int y2;
} Line;

typedef struct Point
{
	int x;
	int y;
} Point;

typedef struct Contour
{
	Point* points;
	int size;
} Contour;

typedef struct Square
{
	Point topLeft;
	Point topRight;
	Point bottomLeft;
	Point bottomRight;
} Square;

typedef struct Intersection
{
	int line2Id;
	Point* point;
} Intersection;

typedef struct DetectionInfo
{
	Square* bestSquare;
	Matrix* e;
	float angle;
} DetectionInfo;

namespace Imagery
{
	Matrix* LoadImageAsMatrix(const QString& imagePath);

	Matrix* ConvertToGrayscale(const Matrix& m);

	Matrix* GetGaussianKernel(int size);

	void Blur(const Matrix& m, Matrix& output, int strength);

	void
	AdaptiveThreshold(const Matrix& m, Matrix& output, int neighborhoodSize, float offset, float highValue,
					  float lowValue);

	void Dilate(const Matrix& m, Matrix& output, int neighborhoodSize);

	void Erode(const Matrix& m, Matrix& output, int neighborhoodSize);

	void BitwiseNot(const Matrix& m, Matrix& output);

	void Canny(const Matrix& m, Matrix& output, float lowThreshold, float highThreshold);

	Matrix** GetGradients(const Matrix& m);

	void NonMaximumSuppression(const Matrix& m, const Matrix& angles, Matrix& output);

	void DoubleThreshold(const Matrix& m, Matrix& output, float lowThreshold, float highThreshold);

	void Hysteresis(const Matrix& m, Matrix& output, float highThreshold);

	Matrix* HoughTransform(const Matrix& m);

	float* kMeansClustersCenter(const int* data, int dataLength);

	float ComputeImageAngle(const HoughLine* lines, int numLines);

	void BilateralFilter(const Matrix& input, Matrix& output, int diameter, float sigma_color, float sigma_space);

	void SobelEdgeDetector(const Matrix& m, Matrix& output);

	void AdaptiveThreshold2(const Matrix& m, Matrix& output, float threshold, float highValue, float lowValue);

	float ComputeEntropy(const Matrix& m);

	float SegmentBlackPercentage(const Matrix& img, const Point* p1, const Point* p2);

	void
	PrintPointScreenCoordinates(const Point* point, int screenWidth, int screenHeight, int imgWidth, int imgHeight);

	float StandardDeviation(const Matrix& m, int blockSize);

	float Brightness(const Matrix& img);

	float Dispersion(const Matrix& in);

	Matrix** CropBorders(const Matrix** cells, int cropPercentage);

	void RemoveBorderArtifacts(Matrix** cells);

	int* GetEmptyCells(const Matrix** cells, float emptinessThreshold);

	Matrix** ResizeCellsTo28x28(const Matrix** cells);

	// Offset pixels horizontally to center the digit. Discards pixels that go out of bounds. Offset can be negative.
	void HorizontalOffset(Matrix& m, int offset);

	void VerticalOffset(Matrix& m, int offset);

	Matrix** CenterCells(const Matrix** cells, const int* emptyCells);

	// Delete a pixel and recursively delete its neighbors if they are white.
	void RemoveContinuousPixels(Matrix& img, int y, int x);

	void RemoveLines(Matrix& img);

	Matrix* Rotation(const Matrix& matrix, const Square& s, double degree);

	void RotatePoint(const Point* pt, const Point* center, float angle, Point* res);

	Square* RotateSquare(const Square* s, const Point* center, float angle);

	float Dist(const Point* pt1, const Point* pt2);

	Point* ClosestEdgeFrom(const Square* s, const Point* pt);

	Square* Order(const Square* s, int w, int h);

	// Function to extract the Sudoku region from the straightened image
	Matrix*
	ExtractSudokuFromStraightImg(const Matrix& straightImage, const Square& sudokuEdges, float rotationAngle);

	Matrix** Split(const Matrix& matrix);

	QPoint PointToQPoint(const Point& p);

	Point QPointToPoint(const QPoint& p);
}

#endif //SUDOKUOCR_IMAGERY_H
