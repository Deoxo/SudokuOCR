//
// Created by mat on 03/10/23.
//

#ifndef S3PROJECT_GRIDDETECTION_H
#define S3PROJECT_GRIDDETECTION_H

#include "Imagery.h"
#include "../Tools/List.h"

namespace GridDetection
{
	HoughLine* FindLines(const Matrix& m, int* numLines);

	Line* HoughLinesToCartesianLines(const HoughLine* houghLines, int numLines, int imgWidth, int imgHeight);

	Square GetDesiredEdges(const Square& sudokuEdges, float angle, int outputSize);

	Matrix*
	ExtractBiggestPixelGroupAndCorners(const Matrix& img, int halfWindowSize, Square* corners, float target = 255.f);

	std::list<std::list<QPoint>> GetPixelGroups(const Matrix& img, int halfWindowSize, float target = 255.f);

	std::list<QPoint> AurelCornerDetection(std::list<QPoint>& points);

	Matrix** ExtractDigits(const Matrix** cells, const bool* emptyCells);
}

#endif //S3PROJECT_GRIDDETECTION_H
