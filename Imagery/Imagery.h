//
// Created by mat on 27/12/23.
//

#ifndef SUDOKUOCR_IMAGERY_H
#define SUDOKUOCR_IMAGERY_H

#include <QString>
#include <QPointF>
#include "Tools/Matrix.h"
#include "../Tools/List.h"
#include <list>
#include <QImage>

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

	Point(int x, int y)
	{
		this->x = x;
		this->y = y;
	}

	Point()
	{
		x = 0;
		y = 0;
	}

	Point(const Point& p)
	{
		x = p.x;
		y = p.y;
	}

	explicit Point(QPoint p)
	{
		x = p.x();
		y = p.y();
	}

	explicit operator QPoint() const
	{
		return {x, y};
	}
} Point;

typedef struct Square
{
	Point topLeft;
	Point topRight;
	Point bottomLeft;
	Point bottomRight;
} Square;

typedef struct DetectionInfo
{
	Matrix* img;
	Square* bestSquare;
	Matrix* e;
	float angle;
} DetectionInfo;

namespace Imagery
{
	Matrix* LoadImageAsMatrix(const QString& imagePath);

	Matrix* ConvertToGrayscale(const Matrix& m);

	Matrix* GetGaussianKernel(int size);

	Matrix* Blur(const Matrix& m, int strength);

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

	float ComputeImageAngle(const HoughLine* lines, int numLines);

	void BilateralFilter(const Matrix& input, Matrix& output, int diameter, float sigma_color, float sigma_space);

	void AdaptiveThreshold2(const Matrix& m, Matrix& output, float threshold, float highValue, float lowValue);

	float SegmentBlackPercentage(const Matrix& img, const Point* p1, const Point* p2);

	float Dispersion(const Matrix& in);

	bool* GetEmptyCells(const Matrix** cells, float emptinessThreshold);

	// Delete a pixel and recursively delete its neighbors if they are white.
	void RemoveContinuousPixels(Matrix& img, int oy, int ox, int halfWindowSize);

	void RemoveLines(Matrix& img);

	void RotatePoint(const Point& pt, const Point& center, float angle, Point& res);

	Matrix** Split(const Matrix& matrix);

	Matrix* BuildPerspectiveMatrix(const Square& sudokuEdges, const Square& desiredEdges);

	Matrix**
	PerspectiveTransform(const Matrix** imgs, const int numImgs, const Square& desiredEdges, const int squareSize,
						 const Square& sudokuEdges);

	void FloodFill(Matrix& img, QPoint pos, int halfWindowSize, std::list<QPoint>& group, float target);

	// Compute the distance between two points
	[[nodiscard]] float Dist(const Point& pt1, const Point& pt2);

	// Compute the distance between two points
	[[nodiscard]] float Dist(const QPoint& a, const QPoint& b);

	Matrix** CenterAndResizeDigits(const Matrix** cells, const bool* emptyCells);

	QImage LoadImg(const QString& path, int maxSize = -1);
}

#endif //SUDOKUOCR_IMAGERY_H
