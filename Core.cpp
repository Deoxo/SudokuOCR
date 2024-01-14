#include "Core.h"
#include "Imagery/GridDetection.h"
#include "Tools/Settings.h"
#include "NeuralNetwork/Network.h"
#include <QDir>
#include <iostream>
#include <QPainter>

Core::Core()
= default;

void DrawLines(const QString& imagePath, const QString& savePath, Line* cartesianLines, const int numLines,
			   HoughLine* lines)
{
	QImage image(imagePath);
	image.convertTo(QImage::Format_RGB32);
	QPainter painter(&image);
	painter.setPen(QPen(Qt::red, 2));
	for (int i = 0; i < numLines; ++i)
	{
		const float angle = lines[i].theta * 180 / M_PI;
		if (angle <= 10)
			painter.setPen(QPen(Qt::blue, 2));
		else if (angle <= 90)
			painter.setPen(QPen(Qt::red, 2));
		else
			painter.setPen(QPen(Qt::green, 2));
		painter.drawLine(cartesianLines[i].x1, cartesianLines[i].y1, cartesianLines[i].x2, cartesianLines[i].y2);	}

	image.save(savePath + "/7-lines.png");
}

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
	StepCompletedWrapper(*blurred, "2-blurred", savePath);

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
	StepCompletedWrapper(*binarized, "3-binarized", savePath);

	// Invert image
	Matrix* inversed = Matrix::CreateSameSize(*m);
	Imagery::BitwiseNot(*binarized, *inversed);
	StepCompletedWrapper(*inversed, "4-inversed", savePath);

	// Dilate
	Matrix* dilated = Matrix::CreateSameSize(*m);
	Imagery::Dilate(*inversed, *dilated, DELATE_KERNEL_SIZE);
	StepCompletedWrapper(*dilated, "5.0-dilated", savePath);

	Matrix* e = Matrix::CreateSameSize(*m);
	if (dispersion >= 500)
		Imagery::Erode(*dilated, *e, DELATE_KERNEL_SIZE);
	else
		dilated->CopyValuesTo(*e);
	StepCompletedWrapper(*e, "5.2-eroded", savePath);

	// Canny
	Matrix* canny = Matrix::CreateSameSize(*dilated);
	Imagery::Canny(*dilated, *canny, 50, 150);
	StepCompletedWrapper(*canny, "6.0-canny", savePath);

	Square* bestSquare = new Square();
	Matrix* mainPixels = Imagery::ExtractBiggestPixelGroupAndCorners(*canny, 3, bestSquare);
	StepCompletedWrapper(*mainPixels, "6.1-mainPixels", savePath);

	// Get edges
	//Matrix* sobelEdges = Matrix::CreateSameSize(*m);
	//Imagery::SobelEdgeDetector(*dilated, *sobelEdges);
	//StepCompletedWrapper(*sobelEdges, "5-sobelEdges", savePath);

	const int sw = dilated->cols, sh = dilated->rows;

	// Get lines
	int numLines = 0;
	HoughLine* lines = GridDetection::FindLines(*mainPixels, &numLines);

	// Convert to cartesian lines
	Line* cartesianLines = GridDetection::HoughLinesToCartesianLines(lines, numLines, sw, sh);
	DrawLines(savePath + "/6.1-mainPixels.png", savePath, cartesianLines, numLines, lines);
	qDebug() << "7-lines";

	const float angle = Imagery::ComputeImageAngle(lines, numLines);
	printf("Angle: %f\n", angle);

    QPoint* vertices = new QPoint[] {
						(QPoint)bestSquare->topLeft,
						(QPoint)bestSquare->bottomRight,
						(QPoint)bestSquare->topRight,
                         (QPoint)bestSquare->bottomLeft
                         };
	emit OnVerticesDetected(vertices);

	DetectionInfo* detectionInfo = new DetectionInfo();

	detectionInfo->bestSquare = bestSquare;
	detectionInfo->e = e;
	detectionInfo->angle = angle;

	// Free memory
	delete img;
	delete m;
	delete bilaterallyFiltered;
	delete dilated0;
	delete eroded0;
	delete blurred;
	delete binarized;
	delete inversed;
	delete dilated;
	//delete sobelEdges;
	delete canny;
	delete mainPixels;
	delete[] lines;
	delete[] cartesianLines;

	return detectionInfo;
}

void Core::DigitDetection(DetectionInfo * detectionInfo, const QString& savePath) const
{
	const int sw = detectionInfo->e->cols, sh = detectionInfo->e->rows;
	const int squareSize = std::min(sw, sh);
	Square desiredSquare = Imagery::GetDesiredEdges(*detectionInfo->bestSquare, -detectionInfo->angle, squareSize - 1);
	qDebug() << "Desired square";

	// Extract the Sudoku from the image
	Matrix* perspective = Imagery::PerspectiveTransform(*detectionInfo->e, *detectionInfo->bestSquare,
														desiredSquare, squareSize);
	perspective->SaveAsImg(savePath, "8-perspective");
	qDebug() << "8-perspective";
	Imagery::RemoveLines(*perspective);
	perspective->SaveAsImg(savePath, "9-removedLines");
	qDebug() << "9-removedLines";
	emit OnDigitsIsolated(savePath + "/9-removedLines");

	// Split the Sudoku into cells
	Matrix** cells = Imagery::Split(*perspective);
	Matrix** borderlessCells = Imagery::CropBorders((const Matrix**) cells, BORDER_CROP_PERCENTAGE);
	Matrix** resizedCells = Imagery::ResizeCellsTo28x28((const Matrix**) borderlessCells);
	int* emptyCells = Imagery::GetEmptyCells((const Matrix**) resizedCells, EMPTY_CELL_THRESHOLD);
	Matrix** centeredCells = Imagery::CenterCells((const Matrix**) resizedCells, emptyCells);

	// Digit recognition
	Matrix* digits = new Matrix(9, 9, 1);
	NeuralNetwork* nn = NeuralNetwork::LoadFromFile("./nn3.bin");
	for (int i = 0; i < 81; i++)
	{
		if (emptyCells[i])
		{
			digits->data[i] = 0;
			continue;
		}
		const QString cellName = "10-" + QString::number(i / 10) + QString::number(i % 10) + "";
		centeredCells[i]->SaveAsImg(savePath, cellName);

		*centeredCells[i] *= 1.f / 255.f;
		digits->data[i] = (float) nn->Predict(*centeredCells[i]) + 1;
	}

	// Free memory
	delete perspective;
	delete nn;
	for (int i = 0; i < 81; ++i)
	{
		delete cells[i];
		delete borderlessCells[i];
		delete centeredCells[i];
		delete resizedCells[i];
	}
	delete[] cells;
	delete[] borderlessCells;
	delete[] centeredCells;
	delete[] resizedCells;
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