//
// Created by mat on 03/10/23.
//

#include "GridDetection.h"
#include "../Tools/Settings.h"
#include "Imagery.h"

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
                    // Remove the line that is the closest to the origin.
                    /*if (line1->rho < line2->rho)
                    {
                        for (int k = i; k < numLines2 - 1; k++)
                            lines[k] = lines[k + 1];
                        numLines2--;
                        j--;
                    }
                    else
                    {
                        for (int k = j; k < numLines2 - 1; k++)
                            lines[k] = lines[k + 1];
                        numLines2--;
                    }*/
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
                    lines[linesIndex].rho = rho - diagonal;
                    lines[linesIndex].theta = theta * M_PI / 180;

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
            if (std::abs(theta) < M_PI / 4.0 || std::abs(theta) > 3.0 * M_PI / 4.0)
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

    float Distance(const Point* pt1, const Point* pt2)
    {
        return std::sqrt(
                std::pow<float, float>(pt1->x - pt2->x, 2) +
                powf(pt1->y - pt2->y, 2)); // NOLINT(*-narrowing-conversions)
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
                const float distance = Distance(pt1, pt2);
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

    List* FindIntersections(Line* cartesianLines, const int numLines, const int imageWidth, const int imgHeight,
                            int* numIntersections)
    {
        List* intersections = ListCreate();

        // Find intersections
        for (int i = 0; i < numLines; i++)
            for (int j = i + 1; j < numLines; j++)
            {
                Line line1 = cartesianLines[i];
                Line line2 = cartesianLines[j];
                const int denominator = (line1.x1 - line1.x2) * (line2.y1 - line2.y2) -
                                        (line1.y1 - line1.y2) * (line2.x1 - line2.x2);

                // If the denominator is 0, the lines are parallel
                if (denominator != 0)
                {
                    // Only consider intersections between perpendicular lines
                    const float theta1 = atan2f(line1.y1 - line1.y2, line1.x1 - line1.x2);
                    const float theta2 = atan2f(line2.y1 - line2.y2, line2.x1 - line2.x2);
                    if (std::abs(theta1 - theta2) < M_PI / 3.f)
                        continue;

                    // Compute intersection point
                    const int x = ((line1.x1 * line1.y2 - line1.y1 * line1.x2) * (line2.x1 - line2.x2) -
                                   (line1.x1 - line1.x2) * (line2.x1 * line2.y2 - line2.y1 * line2.x2)) / denominator;

                    if (x < 0 || x > imageWidth)
                        continue;

                    const int y = ((line1.x1 * line1.y2 - line1.y1 * line1.x2) * (line2.y1 - line2.y2) -
                                   (line1.y1 - line1.y2) * (line2.x1 * line2.y2 - line2.y1 * line2.x2)) / denominator;

                    // Discard points outside the image
                    if (y < 0 || y > imgHeight)
                        continue;

                    // Add the intersection to the list
                    Point* pt = new Point();
                    pt->x = x;
                    pt->y = y;
                    ListAdd(intersections, pt);
                    //printf("Intersection: (%d, %d)\n", x, y);
                    (*numIntersections)++;
                }
            }

        printf("Num intersections: %d\n", *numIntersections);
        FilterIntersections(intersections, numIntersections);
        printf("Num filtered intersections: %d\n", *numIntersections);

        return intersections;
    }

    List* FindIntersections2(Line* cartesianLines, const int numLines, const int imageWidth, const int imgHeight)
    {
        /*if (line1.x1 == line1.x2 && line2.x1 == line2.x2)
                {
                    // Both lines are vertical
                    if (line1.x1 == line2.x1)
                    {
                        // Check if they have the same x-coordinate, which means they overlap.
                        // You can handle this case according to your requirements.
                        continue;
                    }
                    else
                    {
                        // Parallel vertical lines that do not overlap, so no intersection.
                        continue;
                    }
                }
                else if (line1.x1 == line1.x2)
                {
                    // Line 1 is vertical, calculate the intersection using Line 2's equation
                    double m2 = (line2.y2 - line2.y1) / (line2.x2 - line2.x1);
                    int x = line1.x1; // The x-coordinate of the intersection is the same as that of Line 1.
                    int y = m2 * x + (line2.y1 - m2 * line2.x1);
                    if (x < 0 || x > imageWidth)
                        continue;

                    if (y < 0 || y > imgHeight)
                        continue;

                    intersection* inter = malloc(sizeof(intersection));
                    point* pt = malloc(sizeof(point));
                    pt->x = x;
                    pt->y = y;
                    inter->point = pt;
                    inter->line2Id = j;
                    ListAdd(lineIntersections, inter);
                    printf("Intersection: (%d, %d)\n", x, y);
                }
                else if (line2.x1 == line2.x2)
                {
                    // Line 2 is vertical, calculate the intersection using Line 1's equation
                    double m1 = (line1.y2 - line1.y1) / (line1.x2 - line1.x1);
                    int x = line2.x1; // The x-coordinate of the intersection is the same as that of Line 2.
                    int y = m1 * x + (line1.y1 - m1 * line1.x1);
                    if (x < 0 || x > imageWidth)
                        continue;

                    if (y < 0 || y > imgHeight)
                        continue;

                    intersection* inter = malloc(sizeof(intersection));
                    point* pt = malloc(sizeof(point));
                    pt->x = x;
                    pt->y = y;
                    inter->point = pt;
                    inter->line2Id = j;
                    ListAdd(lineIntersections, inter);
                    printf("Intersection: (%d, %d)\n", x, y);
                }
                else
                {
                    // Neither line is vertical, proceed with the general case
                    double m1 = (line1.y2 - line1.y1) / (line1.x2 - line1.x1);
                    double m2 = (line2.y2 - line2.y1) / (line2.x2 - line2.x1);

                    // Check if lines are parallel
                    if (m1 == m2)
                    {
                        continue;
                    }

                    double b1 = line1.y1 - m1 * line1.x1;
                    double b2 = line2.y1 - m2 * line2.x1;

                    int x = (b2 - b1) / (m1 - m2);
                    int y = m1 * x + b1;

                    if (x < 0 || x > imageWidth)
                        continue;

                    if (y < 0 || y > imgHeight)
                        continue;

                    intersection* inter = malloc(sizeof(intersection));
                    point* pt = malloc(sizeof(point));
                    pt->x = x;
                    pt->y = y;
                    inter->point = pt;
                    inter->line2Id = j;
                    ListAdd(lineIntersections, inter);
                    printf("Intersection: (%d, %d)\n", x, y);
                }*/
        List* intersections = ListCreate();

        // Find intersections
        for (int i = 0; i < numLines; i++)
        {
            List* lineIntersections = ListCreate();
            for (int j = 0; j < numLines; j++)
            {
                const Line line1 = cartesianLines[i];
                const Line line2 = cartesianLines[j];

                const float x12 = line1.x1 - line1.x2;
                const float x34 = line2.x1 - line2.x2;
                const float y12 = line1.y1 - line1.y2;
                const float y34 = line2.y1 - line2.y2;
                const float c = x12 * y34 - y12 * x34;
                const float a = line1.x1 * line1.y2 - line1.y1 * line1.x2;
                const float b = line2.x1 * line2.y2 - line2.y1 * line2.x2;
                if (c != 0)
                {
                    // Intersection point
                    const float x = (a * x34 - b * x12) / c;
                    const float y = (a * y34 - b * y12) / c;

                    // Discard points outside the image
                    if (x < 0 || x >= imageWidth)
                        continue;
                    if (y < 0 || y >= imgHeight)
                        continue;

                    // Add the intersection to the list
                    Intersection* inter = new Intersection();
                    Point* pt = new Point();
                    pt->x = x;
                    pt->y = y;
                    inter->point = pt;
                    inter->line2Id = j;
                    ListAdd(lineIntersections, inter);
                    //printf("Intersection: (%f, %f)\n", x, y);
                }

                // Denominator method, seems to not be working all the time.
                /*const int denominator = (line1.x1 - line1.x2) * (line2.y1 - line2.y2) -
                                        (line1.y1 - line1.y2) * (line2.x1 - line2.x2);

                // If the denominator is 0, the lines are parallel
                if (denominator != 0)
                {
                    // Only consider intersections between perpendicular lines
                    const float theta1 = atan2f(line1.y1 - line1.y2, line1.x1 - line1.x2);
                    const float theta2 = atan2f(line2.y1 - line2.y2, line2.x1 - line2.x2);
                    if (std::abs(theta1 - theta2) < M_PI / 3.f)
                        continue;

                    // Compute intersection point
                    const int x = ((line1.x1 * line1.y2 - line1.y1 * line1.x2) * (line2.x1 - line2.x2) -
                                   (line1.x1 - line1.x2) * (line2.x1 * line2.y2 - line2.y1 * line2.x2)) / denominator;

                    //if (x < 0 || x > imageWidth)
                    //    continue;

                    const int y = ((line1.x1 * line1.y2 - line1.y1 * line1.x2) * (line2.y1 - line2.y2) -
                                   (line1.y1 - line1.y2) * (line2.x1 * line2.y2 - line2.y1 * line2.x2)) / denominator;

                    // Discard points outside the image
                    if (y < 0 || y > imgHeight)
                        continue;

                    // Add the intersection to the list
                    intersection* inter = malloc(sizeof(intersection));
                    point* pt = malloc(sizeof(point));
                    pt->x = x;
                    pt->y = y;
                    inter->point = pt;
                    inter->line2Id = j;
                    ListAdd(lineIntersections, inter);
                    printf("Intersection: (%d, %d)\n", x, y);
                }*/
            }
            ListAdd(intersections, lineIntersections);
        }

        return intersections;
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
                Distance(&square->topRight, &square->bottomRight),
                Distance(&square->bottomRight, &square->bottomLeft),
                Distance(&square->bottomLeft, &square->topLeft),
                Distance(&square->topLeft, &square->topRight),
                Distance(&square->topRight, &square->bottomLeft),
                Distance(&square->bottomRight, &square->topLeft),
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
        const float dotProduct1 =
                (square->bottomRight.x - square->topRight.x) * (square->bottomLeft.x - square->bottomRight.x) +
                (square->bottomRight.y - square->topRight.y) * (square->bottomLeft.y - square->bottomRight.y);
        const float dotProduct2 =
                (square->bottomLeft.x - square->bottomRight.x) * (square->topLeft.x - square->bottomLeft.x) +
                (square->bottomLeft.y - square->bottomRight.y) * (square->topLeft.y - square->bottomLeft.y);
        const float dotProduct3 =
                (square->topLeft.x - square->bottomLeft.x) * (square->topRight.x - square->topLeft.x) +
                (square->topLeft.y - square->bottomLeft.y) * (square->topRight.y - square->topLeft.y);
        const float dotProduct4 =
                (square->topRight.x - square->topLeft.x) * (square->bottomRight.x - square->topRight.x) +
                (square->topRight.y - square->topLeft.y) * (square->bottomRight.y - square->topRight.y);

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

    List*
    GetSquares(const List* intersections, const int numIntersections, const float tolerance, int* numSquares)
    {
        List* squares = ListCreate();
        List* lastSquare = squares;

        for (int i = 0; i < numIntersections; ++i)
        {
            const List* li = ListGetList(intersections, i);
            for (int j = i + 1; j < numIntersections; ++j)
            {
                const List* lj = ListGetList(li, j - i - 1);
                for (int k = j + 1; k < numIntersections; ++k)
                {
                    const List* lk = ListGetList(lj, k - j - 1);
                    for (int l = k + 1; l < numIntersections; ++l)
                    {
                        const List* ll = ListGetList(lk, l - k - 1);

                        Square* s = new Square();
                        s->bottomRight = *(Point*) li->data;
                        s->bottomLeft = *(Point*) lj->data;
                        s->topRight = *(Point*) lk->data;
                        s->topLeft = *(Point*) ll->data;
                        if (IsSquare(s, tolerance))
                        {
                            // Add the square to the list
                            ListAdd(lastSquare, s);
                            lastSquare = lastSquare->next;
                            (*numSquares)++;
                        }
                        else delete s;
                    }
                }
            }
        }

        return squares;
    }

    List*
    GetSquares2(const List* intersections, const float tolerance, int* numSquares, const int numLines)
    {
        List* squares = ListCreate();
        List* lastSquare = squares;

        for (int i = 0; i < numLines; ++i)
        {
            List* intersection1_li = ((List*) ListGet(intersections, i))->next;
            while (intersection1_li != nullptr)
            {
                Intersection* inter = (Intersection*) intersection1_li->data;
                Point* pt1 = inter->point;

                List* intersection2_li = ((List*) ListGet(intersections, inter->line2Id))->next;
                while (intersection2_li != nullptr)
                {
                    Intersection* inter2 = (Intersection*) intersection2_li->data;
                    Point* pt2 = inter2->point;
                    if (inter2->line2Id == i)
                        break;

                    List* intersection3_li = ((List*) ListGet(intersections, inter2->line2Id))->next;
                    while (intersection3_li != nullptr)
                    {
                        Intersection* inter3 = (Intersection*) intersection3_li->data;
                        Point* pt3 = inter3->point;

                        if (inter3->line2Id == i || inter3->line2Id == inter->line2Id)
                            break;

                        List* intersection4_li = ((List*) ListGet(intersections, inter3->line2Id))->next;
                        while (intersection4_li != nullptr)
                        {
                            Intersection* inter4 = (Intersection*) intersection4_li->data;
                            Point* pt4 = inter4->point;

                            Square* s = new Square();
                            s->topRight = *pt1;
                            s->bottomRight = *pt2;
                            s->bottomLeft = *pt3;
                            s->topLeft = *pt4;
                            if (i == inter4->line2Id && IsSquare(s, tolerance))
                            {
                                // Add the square to the list
                                ListAdd(lastSquare, s);
                                lastSquare = lastSquare->next;
                                (*numSquares)++;
                            }
                            else delete s;

                            intersection4_li = intersection4_li->next;
                        }

                        intersection3_li = intersection3_li->next;
                    }

                    intersection2_li = intersection2_li->next;
                }

                intersection1_li = intersection1_li->next;
            }
        }

        return squares;
    }

    int SquareFitness(const Square* s, const Matrix& dilated)
    {
        float edgeDistances[6] = {
                Distance(&s->topRight, &s->bottomRight),
                Distance(&s->bottomRight, &s->bottomLeft),
                Distance(&s->bottomLeft, &s->topLeft),
                Distance(&s->topLeft, &s->topRight),
                Distance(&s->topRight, &s->bottomLeft),
                Distance(&s->bottomRight, &s->topLeft),
        };
        InsertionSort(edgeDistances, 6);

        // Percentage of black pixels in the square
        const float p = (Imagery::SegmentBlackPercentage(dilated, &s->topRight, &s->bottomRight) +
                         Imagery::SegmentBlackPercentage(dilated, &s->bottomRight, &s->bottomLeft) +
                         Imagery::SegmentBlackPercentage(dilated, &s->bottomLeft, &s->topLeft) +
                         Imagery::SegmentBlackPercentage(dilated, &s->topLeft, &s->topRight)) / 4.f;

        return (int) (edgeDistances[0] * edgeDistances[0] * p * p);
    }

    Square* FindBestSquare(const List* squares, int numSquares, const Matrix& dilated)
    {
        Square* bestSquare = nullptr;
        const List* current = ListGetList(squares, 0);
        int bestFitness = 0;
        for (int i = 0; i < numSquares; i++)
        {
            Square* s = (Square*) current->data;
            const int fitness = SquareFitness(s, dilated);
            if (fitness > bestFitness)
            {
                bestFitness = fitness;
                bestSquare = s;
            }
            current = current->next;
        }

        return bestSquare;
    }
}

