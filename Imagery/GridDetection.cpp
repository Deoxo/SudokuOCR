//
// Created by mat on 03/10/23.
//

#include <QLineF>
#include <iostream>
#include "GridDetection.h"
#include "../Tools/Settings.h"
#include "Imagery.h"
#include "Tools/Math.h"

namespace GridDetection
{
	// Remove the lines that are too close to each other.
	void FilterLines(HoughLine* lines, int* numLines)
	{
		int numLines2 = *numLines;
		for (int i = 0; i < numLines2; i++)
		{
			HoughLine* line1 = &lines[i];
			for (int j = i + 1; j < numLines2; j++)
			{
				HoughLine* line2 = &lines[j];
				const float distance = std::abs(line1->rho - line2->rho);
				const float thetaDistance = std::abs(line1->theta - line2->theta);
				if ((distance < MIN_LINES_RHO_DIST && thetaDistance < MIN_LINES_THETA_DIST))
				{
					// Compute average line, replace the first one with it and delete the other one
					const float midTheta = (line1->theta + line2->theta) / 2.f;
					const float midRho = (line1->rho + line2->rho) / 2.f;

					for (int k = j; k < numLines2 - 1; k++)
						lines[k] = lines[k + 1];
					numLines2--;
					line1->rho = midRho;
					line1->theta = midTheta;
				}
			}
		}
		*numLines = numLines2;
	}

	float ComputeHoughThreshold(const Matrix* accumulator, const float maxIntensity, const int accumulatorRows,
								const int accumulatorCols, int* numLines)
	{
		int rightThresholdFound = 0, prev = 0;
		float houghThreshold = maxIntensity * .4f;
		do
		{
			// Count the number of lines
			for (int rho = 0; rho < accumulatorRows; rho++)
				for (int theta = 0; theta < accumulatorCols; theta++)
				{
					const float intensity = accumulator->data[rho * accumulator->cols + theta];
					if (intensity > houghThreshold)
						(*numLines)++;
				}

			//printf("Hough threshold: %f - NumLines: %i\n", houghThreshold, *numLines);

			// Adjust the threshold
			if (*numLines < MIN_LINES)
			{
				// Prevents from constantly increasing then decreasing the threshold because the number of lines is too high or too low.
				if (prev == 1)
				{
					// Going back to the previous threshold
					houghThreshold /= 1.1f;
					for (int rho = 0; rho < accumulatorRows; rho++)
						for (int theta = 0; theta < accumulatorCols; theta++)
						{
							const float intensity = accumulator->data[rho * accumulator->cols + theta];
							if (intensity > houghThreshold)
								(*numLines)++;
						}
					rightThresholdFound = 1;
				}
				else // Increase the threshold
				{
					prev = -1;
					houghThreshold *= .9f;
					*numLines = 0;
				}
			}
			else if (*numLines > MAX_LINES)
			{
				// Prevents from constantly increasing then decreasing the threshold because the number of lines is too high or too low.
				if (prev == -1)
				{
					// Going back to the previous threshold
					houghThreshold /= 0.9f;
					for (int rho = 0; rho < accumulatorRows; rho++)
						for (int theta = 0; theta < accumulatorCols; theta++)
						{
							const float intensity = accumulator->data[rho * accumulator->cols + theta];
							if (intensity > houghThreshold)
								(*numLines)++;
						}
					rightThresholdFound = 1;
				}
				else // Decrease the threshold
				{
					prev = 1;
					houghThreshold *= 1.1f;
					*numLines = 0;
				}
			}
			else // numLines is between minLines and maxLines
				rightThresholdFound = 1;
		} while (!rightThresholdFound);

		return houghThreshold;
	}

	HoughLine* FindLines(const Matrix& m, int* numLines)
	{
		// Hough transform
		Matrix* accumulator = Imagery::HoughTransform(m);
		const int accumulatorRows = accumulator->rows;
		const int accumulatorCols = accumulator->cols;

		// Find max intensity of the accumulator.
		float maxIntensity = 0;
		for (int rho = 0; rho < accumulatorRows; rho++)
		{
			for (int theta = 0; theta < accumulatorCols; theta++)
			{
				const float intensity = accumulator->data[rho * accumulator->cols + theta];
				if (intensity > maxIntensity)
					maxIntensity = intensity;
			}
		}

		const float houghThreshold = ComputeHoughThreshold(accumulator, maxIntensity, accumulatorRows, accumulatorCols,
														   numLines);

		// Store the lines
		HoughLine* lines = new HoughLine[*numLines];
		const int diagonal = (int) (sqrt(pow(m.rows, 2) + pow(m.cols, 2)) + 1);
		int linesIndex = 0;
		for (int rho = 0; rho < accumulatorRows; rho++)
		{
			for (int theta = 0; theta < accumulatorCols; theta++)
			{
				const float intensity = accumulator->data[rho * accumulator->cols + theta];
				if (intensity > houghThreshold)
				{
					lines[linesIndex].rho = (float) (rho - diagonal);
					lines[linesIndex].theta = (float) theta * M_PIf / 180.f;

					linesIndex++;
				}
			}
		}

		delete accumulator;

		// Filter the lines
		if (VERBOSE)
			printf("Num lines: %d\n", *numLines);
		FilterLines(lines, numLines);
		if (VERBOSE)
			printf("Num filtered lines: %d\n", *numLines);

		return lines;
	}

	// Computes cartesian line from hough line and make sure the line is within the image
	Line*
	HoughLinesToCartesianLines(const HoughLine* houghLines, const int numLines, const int imgWidth, const int imgHeight)
	{
		Line* cartesianLines = new Line[numLines];

		for (int i = 0; i < numLines; i++)
		{
			const HoughLine* houghLine = &houghLines[i];

			double x1, y1, x2, y2;
			const float theta = houghLine->theta;
			const float rho = houghLine->rho;
			if (std::abs(theta) < M_PIf / 4.0 || std::abs(theta) > 3.0 * M_PIf / 4.0)
			{
				// Line is closer to horizontal, solve for y
				y1 = 0;
				x1 = (rho - y1 * std::sin(theta)) / std::cos(theta);
				y2 = imgHeight - 1;
				x2 = (rho - y2 * std::sin(theta)) / std::cos(theta);
			}
			else
			{
				// Line is closer to vertical, solve for x
				x1 = 0;
				y1 = (rho - x1 * std::cos(theta)) / std::sin(theta);
				x2 = imgWidth - 1;
				y2 = (rho - x2 * std::cos(theta)) / std::sin(theta);
			}

			Line cartesianLine;
			cartesianLine.x1 = (int) x1;
			cartesianLine.y1 = (int) y1;
			cartesianLine.x2 = (int) x2;
			cartesianLine.y2 = (int) y2;
			//printf("Line: (%d, %d) (%d, %d)\n", cartesianLine.x1, cartesianLine.y1, cartesianLine.x2, cartesianLine.y2);

			cartesianLines[i] = cartesianLine;
		}


		return cartesianLines;
	}

	// Remove the lines that are too close to each other.
	void FilterIntersections(List* intersections, int* numIntersections)
	{
		int numIntersections2 = *numIntersections;
		for (int i = 0; i < numIntersections2; i++)
		{
			Point* pt1 = (Point*) ListGet(intersections, i);
			for (int j = i + 1; j < numIntersections2; j++)
			{
				Point* pt2 = (Point*) ListGet(intersections, j);
				const float distance = Imagery::Dist(*pt1, *pt2);
				if (distance < MIN_INTERSECTION_DIST)
				{
					// Calculate the middle point
					const int middleX = (pt1->x + pt2->x) / 2;
					const int middleY = (pt1->y + pt2->y) / 2;

					// Remove one intersection
					ListRemove(intersections, j);
					numIntersections2--;

					// Replace the intersection with the middle point
					pt1->x = middleX;
					pt1->y = middleY;

					j--;
				}
			}
		}

		*numIntersections = numIntersections2;
	}

	void InsertionSort(float* array, const int array_size)
	{
		for (int i = 1; i < array_size; i++)
		{
			float key = array[i];
			int j = i - 1;
			while (j >= 0 && array[j] > key)
			{
				array[j + 1] = array[j];
				j--;
			}
			array[j + 1] = key;
		}
	}

	int isBetween(const float x, const float a, const float b)
	{
		return x >= a && x <= b;
	}

	int IsSquare(const Square* square, const float tolerance)
	{
		// Edge distances
		const float d[6] = {
				Imagery::Dist(square->topRight, square->bottomRight),
				Imagery::Dist(square->bottomRight, square->bottomLeft),
				Imagery::Dist(square->bottomLeft, square->topLeft),
				Imagery::Dist(square->topLeft, square->topRight),
				Imagery::Dist(square->topRight, square->bottomLeft),
				Imagery::Dist(square->bottomRight, square->topLeft),
		};

		// Sorted edge distances
		float distances[6] = {
				d[0],
				d[1],
				d[2],
				d[3],
				d[4],
				d[5],
		};

		// Sort the distances (with insertion sort as it is faster than quicksort for small arrays)
		InsertionSort(distances, 6);

		// Check if the distances are equal (only checking the first 4 as the last 2 are diagonals)
		for (int i = 0; i < 3; i++)
			if (std::abs(distances[i] / distances[i + 1]) < tolerance)
				return 0;

		// Compute dot products
		const float dotProduct1 = (float) (
				(square->bottomRight.x - square->topRight.x) * (square->bottomLeft.x - square->bottomRight.x) +
				(square->bottomRight.y - square->topRight.y) * (square->bottomLeft.y - square->bottomRight.y));
		const float dotProduct2 = (float) (
				(square->bottomLeft.x - square->bottomRight.x) * (square->topLeft.x - square->bottomLeft.x) +
				(square->bottomLeft.y - square->bottomRight.y) * (square->topLeft.y - square->bottomLeft.y));
		const float dotProduct3 = (float) (
				(square->topLeft.x - square->bottomLeft.x) * (square->topRight.x - square->topLeft.x) +
				(square->topLeft.y - square->bottomLeft.y) * (square->topRight.y - square->topLeft.y));
		const float dotProduct4 = (float) (
				(square->topRight.x - square->topLeft.x) * (square->bottomRight.x - square->topRight.x) +
				(square->topRight.y - square->topLeft.y) * (square->bottomRight.y - square->topRight.y));

		// Remap dot products to [-1, 1]
		const float dotProduct1Remapped = dotProduct1 / (d[0] * d[1]);
		const float dotProduct2Remapped = dotProduct2 / (d[1] * d[2]);
		const float dotProduct3Remapped = dotProduct3 / (d[2] * d[3]);
		const float dotProduct4Remapped = dotProduct4 / (d[3] * d[0]);

		// Check if angles are right angles
		if (!isBetween(std::abs(dotProduct1Remapped), .0f, MAX_RIGHT_ANGLE_ABS_DOT_PRODUCT) ||
			!isBetween(std::abs(dotProduct2Remapped), .0f, MAX_RIGHT_ANGLE_ABS_DOT_PRODUCT) ||
			!isBetween(std::abs(dotProduct3Remapped), .0f, MAX_RIGHT_ANGLE_ABS_DOT_PRODUCT) ||
			!isBetween(std::abs(dotProduct4Remapped), .0f, MAX_RIGHT_ANGLE_ABS_DOT_PRODUCT))
			return 0;

		//printf("%f %f %f %f\n", dotProduct1Remapped, dotProduct2Remapped, dotProduct3Remapped, dotProduct4Remapped);

		return 1;
	}

	int SquareFitness(const Square* s, const Matrix& dilated)
	{
		float edgeDistances[6] = {
				Imagery::Dist(s->topRight, s->bottomRight),
				Imagery::Dist(s->bottomRight, s->bottomLeft),
				Imagery::Dist(s->bottomLeft, s->topLeft),
				Imagery::Dist(s->topLeft, s->topRight),
				Imagery::Dist(s->topRight, s->bottomLeft),
				Imagery::Dist(s->bottomRight, s->topLeft),
		};
		InsertionSort(edgeDistances, 6);

		// Percentage of black pixels in the square
		const float p = (Imagery::SegmentBlackPercentage(dilated, &s->topRight, &s->bottomRight) +
						 Imagery::SegmentBlackPercentage(dilated, &s->bottomRight, &s->bottomLeft) +
						 Imagery::SegmentBlackPercentage(dilated, &s->bottomLeft, &s->topLeft) +
						 Imagery::SegmentBlackPercentage(dilated, &s->topLeft, &s->topRight)) / 4.f;

		return (int) (edgeDistances[0] * edgeDistances[0] * p * p);
	}

	// Returns the index of the closest point to the target point
	int ClosestPointIndex(const QPoint points[4], const QPoint& target)
	{
		int index = 0;
		qreal minDist = QLineF(points[0], target).length();
		for (int i = 1; i < 4; ++i)
		{
			const qreal dist = QLineF(points[i], target).length();
			if (dist < minDist)
			{
				index = i;
				minDist = dist;
			}
		}

		return index;
	}

	// Computes the desired edges of the sudoku (ie the edges with the corrected perspective)
	Square GetDesiredEdges(const Square& sudokuEdges, const float angle, const int outputSize)
	{
		// Load points with qt
		QPoint perspectivePoints[4];
		QPoint correctedPoints[4];
		perspectivePoints[0] = (QPoint) sudokuEdges.topLeft;
		perspectivePoints[1] = (QPoint) sudokuEdges.topRight;
		perspectivePoints[2] = (QPoint) sudokuEdges.bottomLeft;
		perspectivePoints[3] = (QPoint) sudokuEdges.bottomRight;

		// Find the center
		QPoint center =
				(perspectivePoints[0] + perspectivePoints[1] + perspectivePoints[2] + perspectivePoints[3]) / 4.0;

		// Rotate all points around the center to make the sudoku edges parallel to the image edges
		for (int i = 0; i < 4; i++)
			correctedPoints[i] = Math::RotateQPoint(perspectivePoints[i], center, angle);

		// Find the closest point to each corner of the image
		const int topLeftIndex = ClosestPointIndex(correctedPoints, {0, 0});
		const int topRightIndex = ClosestPointIndex(correctedPoints, {outputSize, 0});
		const int bottomLeftIndex = ClosestPointIndex(correctedPoints, {0, outputSize});
		const int bottomRightIndex = ClosestPointIndex(correctedPoints, {outputSize, outputSize});

		// Set the desired edges
		correctedPoints[topLeftIndex] = {0, 0};
		correctedPoints[topRightIndex] = {outputSize, 0};
		correctedPoints[bottomLeftIndex] = {0, outputSize};
		correctedPoints[bottomRightIndex] = {outputSize, outputSize};

		// Return the desired edges
		Square res;
		res.topLeft = Point(correctedPoints[0]);
		res.topRight = Point(correctedPoints[1]);
		res.bottomLeft = Point(correctedPoints[2]);
		res.bottomRight = Point(correctedPoints[3]);

		return res;
	}

	Matrix*
	ExtractBiggestPixelGroupAndCorners(const Matrix& img, const int halfWindowSize, Square* corners, const float target)
	{
		Matrix* res = Matrix::CreateSameSize(img);

		// Group with most elements
		std::list<std::list<QPoint>> pixelGroups = GetPixelGroups(img, halfWindowSize, target);

		pixelGroups.sort([](const std::list<QPoint>& a, const std::list<QPoint>& b)
						 {
							 return a.size() > b.size();
						 });

		// Select the group with the largest area which has at least 80% of the pixels of the largest group
		const int maxGroupSize = (int) pixelGroups.front().size();
		float maxArea = 0;
		std::list<QPoint> aurelCorners;
		int bestGroupIndex = 0;
		int c = -1;
		for (auto& pg : pixelGroups)
		{
			c++;
			// Exit if the group is too small (as the groups are sorted by size, the following groups will be smaller, so we break)
			if (pg.size() < maxGroupSize * .8f)
				break;

			std::list<QPoint> groupCorners = AurelCornerDetection(pg);
			Square s;
			s.topLeft = Point(*std::next(groupCorners.begin(), 0));
			s.topRight = Point(*std::next(groupCorners.begin(), 1));
			s.bottomLeft = Point(*std::next(groupCorners.begin(), 2));
			s.bottomRight = Point(*std::next(groupCorners.begin(), 3));

			float edgeDistances[6] = {
					Imagery::Dist(s.topRight, s.bottomRight),
					Imagery::Dist(s.bottomRight, s.bottomLeft),
					Imagery::Dist(s.bottomLeft, s.topLeft),
					Imagery::Dist(s.topLeft, s.topRight),
					Imagery::Dist(s.topRight, s.bottomLeft),
					Imagery::Dist(s.bottomRight, s.topLeft),
			};
			InsertionSort(edgeDistances, 6);

			const float area = edgeDistances[0] * edgeDistances[1];
			if (area > maxArea)
			{
				maxArea = area;
				aurelCorners = groupCorners;
			}
		}

		std::list<QPoint>& mainGroup = *std::next(pixelGroups.begin(), bestGroupIndex);

		// Write the main group to the result matrix
		for (const QPoint& pixel : mainGroup)
			res->data[pixel.y() * res->cols + pixel.x()] = target;

		// Write the corners to the result square
		std::list<QPoint>::iterator it = aurelCorners.begin();
		corners->topLeft = Point(*it++);
		corners->topRight = Point(*it++);
		corners->bottomLeft = Point(*it++);
		corners->bottomRight = Point(*it++);

		return res;
	}

	// Returns the main group of contiguous pixels in the image
	std::list<std::list<QPoint>> GetPixelGroups(const Matrix& img, const int halfWindowSize, const float target)
	{
		Matrix* tmp = Matrix::CreateSameSize(img);
		img.CopyValuesTo(*tmp);
		// Find all pixel groups
		std::list<std::list<QPoint>> pixelGroups;
		for (int y = 0; y < tmp->rows; ++y)
		{
			for (int x = 0; x < tmp->cols; ++x)
			{
				if (tmp->data[y * tmp->cols + x] == 255)
				{
					std::list<QPoint> group;
					Imagery::FloodFill(*tmp, QPoint(x, y), halfWindowSize, group, target);
					pixelGroups.push_back(group);
				}
			}
		}

		delete tmp;
		return pixelGroups;
	}

	// Returns the point that is the farthest from all anchors
	QPoint GetFarthestPointFromAnchors(const std::list<QPoint>& anchors, const std::list<QPoint>& pts)
	{
		int maxIndex = 0;
		float maxDistSum = 0;
		int i = 0;
		for (const QPoint& p : pts)
		{
			float distsSum = 0;
			for (const QPoint& anchor : anchors)
				distsSum += Imagery::Dist(p, anchor);
			if (distsSum >= maxDistSum)
			{
				maxDistSum = distsSum;
				maxIndex = i;
			}
			i++;
		}

		return *std::next(pts.begin(), maxIndex);
	}

	std::list<QPoint> AurelCornerDetection(std::list<QPoint>& points)
	{
		std::list<QPoint> anchors = {points.front()};

		QPoint corner1 = GetFarthestPointFromAnchors(anchors, points);
		anchors.remove(anchors.front());
		points.remove(corner1);
		anchors.push_back(corner1);

		QPoint corner2 = GetFarthestPointFromAnchors(anchors, points);
		points.remove(anchors.front());
		anchors.push_back(corner2);

		QPoint corner3 = GetFarthestPointFromAnchors(anchors, points);
		points.remove(anchors.front());
		anchors.push_back(corner3);

		QPoint corner4 = GetFarthestPointFromAnchors(anchors, points);
		points.remove(anchors.front());
		anchors.push_back(corner4);

		return anchors;
	}


	Matrix** ExtractDigits(const Matrix** cells, const bool* emptyCells)
	{
		Matrix** digits = new Matrix* [81];
		const int halfWindowSize = (int) ((float) std::min(cells[0]->cols, cells[0]->rows) / 14.f);
		const QPoint cellCenter(cells[0]->cols / 2, cells[0]->rows / 2);
		for (int i = 0; i < 81; ++i)
		{
			digits[i] = Matrix::CreateSameSize(*cells[i]);

			// Skip empty cells
			if (emptyCells[i])
				continue;

			// Get pixel groups
			std::list<std::list<QPoint>> pixelGroups = GetPixelGroups(*cells[i], halfWindowSize);

			// Extract the pixel group of the digit based on the fitness | fitness(group) = size * distRatio^3
			// distRatio = 1 - avgDistFromCenter / maxDistFromCenter

			const int n = (int) pixelGroups.size(); // Number of groups
			float groupsAvgCenterDist[n];
			int groupScores[n];
			int groupID = 0;
			for (const auto& pixelGroup : pixelGroups)
			{
				// Compute the average distance from the center of the cell
				groupsAvgCenterDist[groupID] = 0;
				for (const auto& p : pixelGroup)
					groupsAvgCenterDist[groupID] += Imagery::Dist(p, cellCenter);
				groupsAvgCenterDist[groupID] /= (float) pixelGroup.size();

				// Compute distRatio
				const float distRatio = 1 - groupsAvgCenterDist[groupID] / Imagery::Dist(QPoint(0, 0), cellCenter);

				//Compute fitness
				groupScores[groupID] = (int) ((float) pixelGroup.size() * distRatio * distRatio * distRatio);

				groupID++;
			}

			// Find best group
			int bestGroupID = 0;
			int bestGroupScore = groupScores[0];
			for (int j = 1; j < n; ++j)
			{
				if (groupScores[j] > bestGroupScore)
				{
					bestGroupID = j;
					bestGroupScore = groupScores[j];
				}
			}

			// Write the best group to the result matrix
			for (const auto& p : *std::next(pixelGroups.begin(), bestGroupID))
				digits[i]->data[p.y() * digits[i]->cols + p.x()] = 255.f;
		}

		return digits;
	}
}

