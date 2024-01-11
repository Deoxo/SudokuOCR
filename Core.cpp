#include "Core.h"
#include "Imagery/GridDetection.h"
#include "Tools/Settings.h"
#include "NeuralNetwork/Network.h"
#include <QDir>
#include <iostream>
#include <QPainter>

Core::Core()
= default;

DetectionInfo* Core::BordersDetection(const QString& imagePath, const QString& savePath) const
{
	// Convert the image to grayscale and stores it in a matrix
	Matrix* img = Imagery::LoadImageAsMatrix(imagePath);
	Matrix* m = Imagery::ConvertToGrayscale(*img);
	StepCompletedWrapper(*m, "0-grayscale", savePath);
	//printf("Standard deviation: %f\n", StandardDeviation(m, 5));

	// Noise reduction
	Matrix* bilaterallyFiltered = Matrix::CreateSameSize(*m);
	Imagery::BilateralFilter(*m, *bilaterallyFiltered, BIL_FIL_DIAM, BIL_FIL_SIG_COL, BIL_FIL_SIG_SPACE);
	StepCompletedWrapper(*bilaterallyFiltered, "1.0-bilaterallyFiltered", savePath);

	// Opening
	// Dilate
	Matrix* dilated0 = Matrix::CreateSameSize(*m);
	Imagery::Erode(*bilaterallyFiltered, *dilated0, DELATE_KERNEL_SIZE);
	StepCompletedWrapper(*dilated0, "1.1-dilated", savePath);

	// Erode
	Matrix* eroded0 = Matrix::CreateSameSize(*m);
	Imagery::Dilate(*dilated0, *eroded0, DELATE_KERNEL_SIZE);
	StepCompletedWrapper(*eroded0, "1.2-eroded", savePath);

	// Blur
	Matrix* blurred = Matrix::CreateSameSize(*m);
	Imagery::Blur(*eroded0, *blurred, 5);
	StepCompletedWrapper(*blurred, "1.3-blurred", savePath);

	// Binarize
	Matrix* binarized = Matrix::CreateSameSize(*m);
	clock_t start = clock();
	const float dispersion = Imagery::Dispersion(*m);
	printf("%sDispersion: %f\n%s", GREEN, dispersion, RESET);
	// Choose binarization depending on image's dispersion
	if (dispersion < 500)
	{
		printf("%sUsing adaptive threshold 1\n%s", BLUE, RESET);
		Imagery::AdaptiveThreshold(*blurred, *binarized, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_STRENGTH,
								   ADAPTIVE_THRESHS_HIGH, ADAPTIVE_THRESHS_LOW);
	}
	else
	{
		printf("%sUsing adaptive threshold 2\n%s", BLUE, RESET);
		Imagery::AdaptiveThreshold2(*blurred, *binarized, dispersion / 3414.f * ADAPTIVE_THRESH2_THRESHOLD,
									ADAPTIVE_THRESHS_HIGH,
									ADAPTIVE_THRESHS_LOW);
	}
	if (VERBOSE)
	{
		clock_t end = clock();
		printf("Adaptive threshold took %f seconds\n", (double) (end - start) / CLOCKS_PER_SEC);
	}
	StepCompletedWrapper(*binarized, "2-binarized", savePath);

	// Invert image
	Matrix* inversed = Matrix::CreateSameSize(*m);
	Imagery::BitwiseNot(*binarized, *inversed);
	StepCompletedWrapper(*inversed, "3-inversed", savePath);

	// Dilate
	Matrix* dilated = Matrix::CreateSameSize(*m);
	Imagery::Dilate(*inversed, *dilated, DELATE_KERNEL_SIZE);
	StepCompletedWrapper(*dilated, "4.0-dilated", savePath);

	Matrix* e = Matrix::CreateSameSize(*m);
	if (dispersion >= 500)
		Imagery::Erode(*dilated, *e, DELATE_KERNEL_SIZE);
	else
		dilated->CopyValuesTo(*e);
	StepCompletedWrapper(*e, "4.1-eroded", savePath);

	// Canny
	//Matrix* canny = MatCreateSameSize(dilated);x
	//Canny(dilated, canny, 50, 150);

	// Get edges
	Matrix* sobelEdges = Matrix::CreateSameSize(*m);
	Imagery::SobelEdgeDetector(*dilated, *sobelEdges);
	StepCompletedWrapper(*sobelEdges, "5-sobelEdges", savePath);

	qDebug() << "delete";
	delete bilaterallyFiltered;
	delete dilated0;
	delete eroded0;
	delete blurred;
	delete binarized;
	delete inversed;
	delete sobelEdges;
	//MatFree(dilated);
	//MatFree(canny);

	qDebug() << "end";

	const int sw = dilated->cols, sh = dilated->rows;

	// Get lines
	int numLines = 0;
	HoughLine* lines = GridDetection::FindLines(*dilated, &numLines);
	const float angle = Imagery::ComputeImageAngle(lines, numLines);
	printf("Angle: %f\n", angle);

	// Convert to cartesian lines
	Line* cartesianLines = GridDetection::HoughLinesToCartesianLines(lines, numLines, sw, sh);

	//printf("Angle: %s%i%s\n", MAGENTA, ComputeImageAngle(lines, numLines), RESET);

	// Get intersections
	List* intersections = GridDetection::FindIntersections2(cartesianLines, numLines, sw, sh);

	// Get squares
	int numSquares = 0;
	List* squares = GridDetection::GetSquares2(intersections, SQUARES_EDGE_DIFF_TOL, &numSquares, numLines);
	if (VERBOSE)
		printf("Num squares: %i\n", numSquares);

	// Get the borders of the Sudoku from the squares
	clock_t start2 = clock();
	Square* bestSquare = GridDetection::FindBestSquare(squares, numSquares, *dilated);
	if (bestSquare == nullptr)
		throw std::runtime_error("No square found");
	clock_t end2 = clock();

	if (VERBOSE)
	{
		printf("Find best square took %f seconds\n", (double) (end2 - start2) / CLOCKS_PER_SEC);
		printf("Image dimensions: %ix%i\n", sw, sh);
		// print four edges
		printf("Largest square:\n");
		printf("Top right: ");
		Imagery::PrintPointScreenCoordinates(&bestSquare->topRight, sw, sh, sw, sh);
		printf("Bottom right: ");
		Imagery::PrintPointScreenCoordinates(&bestSquare->bottomRight, sw, sh, sw, sh);
		printf("Bottom left: ");
		Imagery::PrintPointScreenCoordinates(&bestSquare->bottomLeft, sw, sh, sw, sh);
		printf("Top left: ");
		Imagery::PrintPointScreenCoordinates(&bestSquare->topLeft, sw, sh, sw, sh);
	}
    QPoint* vertices = new QPoint[] {Imagery::PointToQPoint(bestSquare->topRight),
                         Imagery::PointToQPoint(bestSquare->bottomRight),
                         Imagery::PointToQPoint(bestSquare->bottomLeft),
                         Imagery::PointToQPoint(bestSquare->topLeft)};
	emit OnVerticesDetected(vertices);

	DetectionInfo* detectionInfo = new DetectionInfo();

	detectionInfo->bestSquare = bestSquare;
	detectionInfo->e = e;
	detectionInfo->angle = angle;

	// Free memory
	delete img;
	delete m;
	delete dilated;
	delete[] lines;
	delete[] cartesianLines;
	delete intersections;
	delete squares;

	return detectionInfo;
}

void Core::DigitDetection(DetectionInfo * detectionInfo, const QString& savePath) const
{
	Matrix* rotated = Imagery::Rotation(*detectionInfo->e, *detectionInfo->bestSquare, -detectionInfo->angle);
	rotated->SaveAsImg(savePath, "6-rotated");
	Matrix* cropped = Imagery::ExtractSudokuFromStraightImg(*rotated, *detectionInfo->bestSquare, -detectionInfo->angle);
	cropped->SaveAsImg(savePath, "7-cropped");
	Imagery::RemoveLines(*cropped);
	Matrix** cells = Imagery::Split(*cropped);
	Matrix** borderlessCells = Imagery::CropBorders((const Matrix**) cells, BORDER_CROP_PERCENTAGE);
	Matrix** resizedCells = Imagery::ResizeCellsTo28x28((const Matrix**) borderlessCells);
	int* emptyCells = Imagery::GetEmptyCells((const Matrix**) resizedCells, EMPTY_CELL_THRESHOLD);
	Matrix** centeredCells = Imagery::CenterCells((const Matrix**) resizedCells, emptyCells);
	Matrix* digits = new Matrix(9, 9, 1);
	NeuralNetwork* nn = NeuralNetwork::LoadFromFile("./nn3.bin");
	for (int i = 0; i < 81; i++)
	{
		if (emptyCells[i])
		{
			digits->data[i] = 0;
			continue;
		}
		const QString cellName = "9-" + QString::number(i / 10) + QString::number(i % 10) + "";
		centeredCells[i]->SaveAsImg(savePath, cellName);

		*centeredCells[i] *= 1.f / 255.f;
		digits->data[i] = (float) nn->Predict(*centeredCells[i]) + 1;
	}

	// Free memory
	delete nn;
	for (int i = 0; i < 81; ++i)
	{
		delete cells[i];
		delete borderlessCells[i];
	}
	delete cells;
	delete borderlessCells;
	delete resizedCells;
	delete emptyCells;

	emit OnDigitsRecognized(digits);
}

void Core::StepCompletedWrapper(const Matrix& img, const QString& stepName, const QString& savePath) const
{
	img.SaveAsImg(savePath, stepName);
	emit StepCompleted(stepName);
}

void Core::SaveSudokuImage(const Matrix& sudokuMatrix, const int size, const QString& filePath)
{
	// Adjust the image size to fit the larger borders and lines
	int imageSize = size + 5; // Increase the size by 10 pixels for larger borders

	// Create an image with the adjusted size
	QImage sudokuImage(imageSize, imageSize, QImage::Format_RGB32);
	sudokuImage.fill(Qt::white); // Set background color to white

	// Create a QPainter to draw on the image
	QPainter painter(&sudokuImage);
	painter.setRenderHint(QPainter::Antialiasing, true);

	// Set font properties
	QFont font("Arial", 20);
	painter.setFont(font);

	// Calculate cell size based on the image dimensions and the number of rows and columns
	int cellSize = size / sudokuMatrix.cols;

	// Calculate block size (3x3 block)
	int blockSize = size / 3;

	// Draw larger borders on the outside
	painter.drawRect(0, 0, imageSize - 1, imageSize - 1);

	// Loop through the Sudoku matrix and draw digits, borders, and lines
	for (int row = 0; row < sudokuMatrix.rows; ++row)
	{
		for (int col = 0; col < sudokuMatrix.cols; ++col)
		{
			int digit = static_cast<int>(sudokuMatrix.data[row * sudokuMatrix.cols + col]);

			// Calculate the position of the cell
			int x = col * cellSize + 5; // Offset by 5 pixels for larger borders
			int y = row * cellSize + 5; // Offset by 5 pixels for larger borders

			// Draw borders between blocks
			if (col % 3 == 0 && col > 0)
				painter.drawRect(x, y - 1, 1, blockSize + 1);

			if (row % 3 == 0 && row > 0)
				painter.drawRect(x - 1, y, blockSize + 1, 1);

			// Draw borders around each cell
			painter.drawRect(x, y, cellSize, cellSize);

			// Skip drawing zeros (empty cells)
			if (digit != 0)
			{
				// Calculate the position of the digit in the cell
				int digitX = x + cellSize * 2 / 5;
				int digitY = y + cellSize * 2 / 3;

				// Draw the digit
				painter.drawText(digitX, digitY, QString::number(digit));
			}
		}
	}

	// Outside border
	painter.setPen(QPen(Qt::black, 10));
	painter.drawRect(0, 0, imageSize - 1, imageSize - 1);

	// Save the image to the specified file path
	if (!sudokuImage.save(filePath))
	{
		// Handle the case where saving fails
		qDebug() << "Error: Unable to save the Sudoku image.";
	}
}