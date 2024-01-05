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

	List*
	FindIntersections(Line* cartesianLines, int numLines, int imageWidth, int imgHeight,
					  int* numIntersections);

	List*
	FindIntersections2(Line* cartesianLines, int numLines, int imageWidth, int imgHeight);

	List*
	GetSquares(const List* intersections, int numIntersections, float tolerance, int* numSquares);

	List*
	GetSquares2(const List* intersections, float tolerance, int* numSquares, int numLines);

	Line* HoughLinesToCartesianLines(const HoughLine* houghLines, int numLines, int imgWidth, int imgHeight);

	Square* FindBestSquare(const List* squares, int numSquares, const Matrix& dilated);
}

#endif //S3PROJECT_GRIDDETECTION_H
