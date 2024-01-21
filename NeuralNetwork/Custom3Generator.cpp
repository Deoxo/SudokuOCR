#include <err.h>
#include "Imagery/Imagery.h"
#include "Imagery/GridDetection.h"
#include "Tools/Settings.h"
#include "Network.h"
#include <QDebug>
#include <iostream>

char* FileName(const char* path)
{
	int fileLen = 0;
	const int pathLen = (int) strlen(path);
	int i = pathLen - 1;
	while (path[i] != '/' && i >= 0)
	{
		fileLen++;
		i--;
	}
	int extensionLen = 1;
	i = pathLen - 1;
	while (path[i] != '.' && i >= 0)
	{
		extensionLen++;
		i--;
	}
	char* fileName = (char*) malloc(sizeof(char) * (fileLen - extensionLen + 1));
	strncpy(fileName, path + pathLen - fileLen, fileLen - extensionLen);
	fileName[fileLen] = '\0';
	return fileName;
}

// Removes all the files in a folder but not the folder itself
void ClearFolder(const char* path)
{
	char* command = (char*) malloc(sizeof(char) * (strlen(path) + 9));
	strcpy(command, "rm -rf ");
	strcat(command, path);
	strcat(command, "*");
	if (system(command) != 0)
		errx(EXIT_FAILURE, "Error while clearing folder %s", path);
	free(command);
}

DetectionInfo* BordersDetection(const QString& imagePath, const QString& savePath)
{
	// Convert the image to grayscale and stores it in a matrix
	Matrix* img = Imagery::LoadImageAsMatrix(imagePath);
	Matrix* m = Imagery::ConvertToGrayscale(*img);

	const int mi = std::min(img->cols, img->rows);
	// Noise reduction
	Matrix* bilaterallyFiltered = Matrix::CreateSameSize(*m);
	Imagery::BilateralFilter(*m, *bilaterallyFiltered, (int) ((float) mi / 500.f * BIL_FIL_DIAM), BIL_FIL_SIG_COL,
							 BIL_FIL_SIG_SPACE);

	// Opening
	// Dilate
	Matrix* dilated0 = Matrix::CreateSameSize(*m);
	Imagery::Erode(*bilaterallyFiltered, *dilated0, DELATE_KERNEL_SIZE);

	// Erode
	Matrix* eroded0 = Matrix::CreateSameSize(*m);
	Imagery::Dilate(*dilated0, *eroded0, DELATE_KERNEL_SIZE);

	// Blur
	Matrix* blurred = Imagery::Blur(*eroded0, GAUSS_BLUR_SIZE);

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

	// Invert image
	Matrix* inversed = Matrix::CreateSameSize(*m);
	Imagery::BitwiseNot(*binarized, *inversed);

	// Dilate
	Matrix* dilated = Matrix::CreateSameSize(*m);
	Imagery::Dilate(*inversed, *dilated, DELATE_KERNEL_SIZE);

	Matrix* e = Matrix::CreateSameSize(*m);
	if (dispersion >= 500)
		Imagery::Erode(*dilated, *e, DELATE_KERNEL_SIZE);
	else
		dilated->CopyValuesTo(*e);

	// Canny
	Matrix* canny = Matrix::CreateSameSize(*dilated);
	Imagery::Canny(*dilated, *canny, CANNY_THRESHOLD1, CANNY_THRESHOLD2);

	Square* bestSquare = new Square();
	Matrix* mainPixels = GridDetection::ExtractBiggestPixelGroupAndCorners(*canny, 3, bestSquare);

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
	qDebug() << "7-lines";

	const float angle = Imagery::ComputeImageAngle(lines, numLines);
	printf("Angle: %f\n", angle);

	if (angle != angle)
	{
		delete img;
		delete m;
		delete bilaterallyFiltered;
		delete dilated0;
		delete eroded0;
		delete blurred;
		delete binarized;
		delete inversed;
		delete dilated;
		delete canny;
		delete e;
		delete mainPixels;
		delete[] lines;
		delete[] cartesianLines;

		return nullptr;
	}

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

void DigitDetection(DetectionInfo* detectionInfo, const QString& savePath, const int imgId, int* imgDigits)
{
	const int sw = detectionInfo->e->cols, sh = detectionInfo->e->rows;
	const int squareSize = std::min(sw, sh);
	Square desiredSquare = GridDetection::GetDesiredEdges(*detectionInfo->bestSquare, -detectionInfo->angle,
														  squareSize - 1);
	qDebug() << "Desired square";

	// Extract the Sudoku from the image
	const Matrix** tmp = new const Matrix* [1]{detectionInfo->e};
	Matrix** p = Imagery::PerspectiveTransform(tmp, 1,
											   desiredSquare, squareSize, *detectionInfo->bestSquare);
	Matrix* perspective = p[0];
	qDebug() << "8-perspective";
	Imagery::RemoveLines(*perspective);
	qDebug() << "9-removedLines";

	// Split the Sudoku into cells
	Matrix** cells = Imagery::Split(*perspective);
	const bool* emptyCells = Imagery::GetEmptyCells((const Matrix**) cells, EMPTY_CELL_THRESHOLD);
	Matrix** cellsDigits0 = GridDetection::ExtractDigits((const Matrix**) cells, emptyCells);
	Matrix** cellsDigits = Imagery::CenterAndResizeDigits((const Matrix**) cellsDigits0, emptyCells);

	// Digit recognition
	for (int i = 0; i < 81; i++)
	{
		if (emptyCells[i])
			continue;

		char* cellName = (char*) malloc(sizeof(char) * 40);
		sprintf(cellName, "digit_%i_%i_%i.png", imgDigits[i], imgId, i);
		cellsDigits[i]->SaveAsImg(savePath, cellName);
	}

	// Free memory
	delete perspective;
	for (int i = 0; i < 81; ++i)
	{
		delete cells[i];
		delete cellsDigits0[i];
		delete cellsDigits[i];
	}
	delete[] p;
	delete[] cells;
	delete[] cellsDigits0;
	delete[] cellsDigits;
	delete emptyCells;
}


int countLines(const char* filename)
{
	FILE* file;
	int count = 0;
	char c;

	file = fopen(filename, "r"); // Open the file in read mode

	if (file == nullptr)
	{
		printf("Unable to open the file.\n");
		return -1; // Return -1 to indicate an error
	}

	// Count the number of lines
	while ((c = fgetc(file)) != EOF)
	{
		if (c == '\n')
		{
			count++;
		}
	}

	// Increment count if the file doesn't end with a newline but contains text
	if (count > 0)
	{
		count++;
	}

	fclose(file); // Close the file

	return count;
}

/*int main(int argc, char** argv)
{
	// Builds the folders to save the results
	char savePath[] = "../datasets/custom3/";
	ClearFolder(savePath);

	int i = 0;
	FILE* file;
	char line[300];

	file = fopen("../datasets/custom2/sudokus.txt", "r"); // Open the file in read mode

	if (file == NULL)
	{
		printf("Unable to open sudokus file.\n");
		return 1;
	}

	// Read the file line by line
	while (fgets(line, sizeof(line), file) != NULL)
	{
		char* spacePosition = strchr(line, ' '); // Find the first space
		// Calculate the length before the space
		int lengthBeforeSpace = spacePosition - line;

		// Extract characters before the space
		char filePath[lengthBeforeSpace + 1]; // +1 for null-terminator
		strncpy(filePath, line, lengthBeforeSpace);
		filePath[lengthBeforeSpace] = '\0';
		int digits[81];
		spacePosition++;
		for (int j = 0; j < 81; ++j)
		{
			digits[j] = *spacePosition - '0';
			spacePosition++;
		}

		DetectionInfo* detection = BordersDetection(filePath, savePath);
		if (detection != nullptr)
		{
			DigitDetection(detection, savePath, i, digits);
			delete detection->e;
			delete detection->bestSquare;
			delete detection;
		}
		i++;
	}

	return EXIT_SUCCESS;
}*/
