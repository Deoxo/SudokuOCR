#include "Core.h"
#include "Imagery/GridDetection.h"
#include "Tools/Settings.h"
#include "NeuralNetwork/Network.h"
#include "Tools/Math.h"
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
		const float angle = (float)lines[i].theta * 180.f / M_PIf;
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

	const int mi = std::min(img->cols, img->rows);

	// Noise reduction
	Matrix* bilaterallyFiltered = Matrix::CreateSameSize(*m);
	Imagery::BilateralFilter(*m, *bilaterallyFiltered, (int)((float)mi / 500.f * BIL_FIL_DIAM), BIL_FIL_SIG_COL * 4, BIL_FIL_SIG_SPACE * 4);
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
	Matrix* blurred = Imagery::Blur(*eroded0, GAUSS_BLUR_SIZE);
	StepCompletedWrapper(*blurred, "2-blurred", savePath);

	// Binarize
	Matrix* binarized = Matrix::CreateSameSize(*m);
	clock_t start = clock();
	const float dispersion = Imagery::Dispersion(*m);
	printf("%sDispersion: %f\n%s", GREEN, dispersion, RESET);
	// Choose binarization depending on image's dispersion
	if (dispersion < 450)
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
	Imagery::Canny(*dilated, *canny, CANNY_THRESHOLD1, CANNY_THRESHOLD2);
	StepCompletedWrapper(*canny, "6.0-canny", savePath);

	Square* bestSquare = new Square();
	Matrix* mainPixels = GridDetection::ExtractBiggestPixelGroupAndCorners(*canny, 3, bestSquare);
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

	detectionInfo->img = img;
	detectionInfo->bestSquare = bestSquare;
	detectionInfo->e = e;
	detectionInfo->angle = angle;

	// Free memory
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

void Core::DigitDetection(DetectionInfo * detectionInfo, const QString& savePath)
{
	const int sw = detectionInfo->e->cols, sh = detectionInfo->e->rows;
	const int squareSize = std::min(sw, sh);
	Square desiredSquare = GridDetection::GetDesiredEdges(*detectionInfo->bestSquare, -detectionInfo->angle, squareSize - 1);
	qDebug() << "Desired square";

	// Extract the Sudoku from the image
	perspectiveMatrix = Imagery::BuildPerspectiveMatrix(*detectionInfo->bestSquare, desiredSquare);
	inversePerspectiveMatrix = Math::Get3x3MatrixInverse(*perspectiveMatrix);
	Matrix* perspective0 = Imagery::PerspectiveTransform(*detectionInfo->img, squareSize, *inversePerspectiveMatrix);
	Matrix* perspective1 = Imagery::PerspectiveTransform(*detectionInfo->e, squareSize, *inversePerspectiveMatrix);
	perspective0->SaveAsImg(savePath, "8-perspective0");
	perspective1->SaveAsImg(savePath, "8-perspective1");
	qDebug() << "8-perspective";
	Imagery::RemoveLines(*perspective1);
	perspective1->SaveAsImg(savePath, "9-removedLines");
	qDebug() << "9-removedLines";
	emit OnDigitsIsolated(savePath + "8-perspective0");

	// Split the Sudoku into cells
	Matrix** cells = Imagery::Split(*perspective1);
	const bool* emptyCells = Imagery::GetEmptyCells((const Matrix**) cells, EMPTY_CELL_THRESHOLD);
	Matrix** cellsDigits0 = GridDetection::ExtractDigits((const Matrix**) cells, emptyCells);
	for (int i = 0; i < 81; ++i)
		if (!emptyCells[i])
			cellsDigits0[i]->SaveAsImg(savePath, "10-" + QString::number(i / 10) + QString::number(i % 10));
	Matrix** cellsDigits = Imagery::CenterAndResizeDigits((const Matrix**) cellsDigits0, emptyCells, &avgDigitCellOffset);
	for (int i = 0; i < 81; ++i)
		if (!emptyCells[i])
			cellsDigits[i]->SaveAsImg(savePath, "11-" + QString::number(i / 10) + QString::number(i % 10));

	// Digit recognition
	Matrix* digits = new Matrix(9, 9);
	NeuralNetwork* nn = NeuralNetwork::LoadFromFile("./nn3.bin");
	for (int i = 0; i < 81; i++)
	{
		if (emptyCells[i])
		{
			digits->data[i] = 0;
			continue;
		}

		*cellsDigits[i] /= 255.f;
		digits->data[i] = (float) nn->Predict(*cellsDigits[i]) + 1;
	}

	// Free memory
	delete perspective0;
	delete perspective1;
	delete nn;
	for (int i = 0; i < 81; ++i)
	{
		delete cells[i];
		delete cellsDigits0[i];
		delete cellsDigits[i];
	}
	delete[] cells;
	delete[] cellsDigits0;
	delete[] cellsDigits;
	//delete[] resizedCells;
	delete emptyCells;
	delete detectionInfo->bestSquare;
	delete detectionInfo->e;
	delete detectionInfo->img;

	emit OnDigitsRecognized(digits);
}

void Core::StepCompletedWrapper(const Matrix& img, const QString& stepName, const QString& savePath) const
{
	img.SaveAsImg(savePath, stepName);
	emit StepCompleted(stepName);
}

void Core::SaveSudokuImage(const Matrix& sudokuMatrix, const QString& imgPath, const QString& savePath,
						   const QString& perspective0Path)
{
	const QImage perspective0 = Imagery::LoadImg(perspective0Path, MAX_IMG_SIZE);

	// Create a QPainter to draw on the image
	QImage straightRes(perspective0.size(), QImage::Format_RGB32);
	QPainter painter(&straightRes);
	straightRes.fill(Qt::white);
	painter.setRenderHint(QPainter::Antialiasing, true);

	// Calculate cell size based on the image dimensions and the number of rows and columns
	const int cellSize = perspective0.width() / sudokuMatrix.cols;

	// Set font properties
	QFont font("Arial", cellSize / 2);
	painter.setFont(font);
	QPen pen(Qt::red);
	pen.setWidth(2);
	painter.setPen(pen);
	painter.setBrush(Qt::red);

	// Loop through the Sudoku matrix and draw digits, borders, and lines
	for (int row = 0; row < sudokuMatrix.rows; ++row)
	{
		for (int col = 0; col < sudokuMatrix.cols; ++col)
		{
			const int digit = static_cast<int>(sudokuMatrix.data[row * sudokuMatrix.cols + col]);
			if (digit == 0)
				continue;

			// Calculate the position of the cell
			const int digitX = col * cellSize + cellSize * 1 / 4 + avgDigitCellOffset.x;
			const int digitY = row * cellSize + cellSize * 3 / 5 + avgDigitCellOffset.y;

			// Draw the digit
			painter.drawText(digitX, digitY, QString::number(digit));
		}
	}

	QImage perspectiveRes = Imagery::LoadImg(imgPath, MAX_IMG_SIZE);
	QRgb white = qRgb(255, 255, 255);
	for (int y = 0; y < perspectiveRes.height(); ++y)
	{
		for (int x = 0; x < perspectiveRes.width(); ++x)
		{
			Point p = Imagery::ApplyPerspectiveTransformation(*perspectiveMatrix, {x,y});
			if (p.x < 0 || p.x >= straightRes.width() || p.y < 0 || p.y >= straightRes.height())
				continue;
			if (straightRes.pixel(p.x, p.y) != white)
				perspectiveRes.setPixel(x, y, straightRes.pixel(p.x, p.y));
		}
	}

	// Save the image to the specified file path
	if (!perspectiveRes.save(savePath))
	{
		// Handle the case where saving fails
		qDebug() << "Error: Unable to save the Sudoku image.";
	}
}

Core::~Core()
{
	delete perspectiveMatrix;
	delete inversePerspectiveMatrix;
}
