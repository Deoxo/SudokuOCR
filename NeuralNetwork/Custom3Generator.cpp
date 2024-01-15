#include <err.h>
#include <sys/stat.h>
#include <dirent.h>
#include "Imagery/Imagery.h"
#include "Tools/List.h"
#include "Imagery/GridDetection.h"
#include "Tools/Settings.h"
#include "Network.h"
#include <QDebug>

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
	//printf("Standard deviation: %f\n", StandardDeviation(m, 5));

	// Noise reduction
	Matrix* bilaterallyFiltered = Matrix::CreateSameSize(*m);
	Imagery::BilateralFilter(*m, *bilaterallyFiltered, BIL_FIL_DIAM, BIL_FIL_SIG_COL, BIL_FIL_SIG_SPACE);

	// Opening
	// Dilate
	Matrix* dilated0 = Matrix::CreateSameSize(*m);
	Imagery::Erode(*bilaterallyFiltered, *dilated0, DELATE_KERNEL_SIZE);

	// Erode
	Matrix* eroded0 = Matrix::CreateSameSize(*m);
	Imagery::Dilate(*dilated0, *eroded0, DELATE_KERNEL_SIZE);

	// Blur
	Matrix* blurred = Matrix::CreateSameSize(*m);
	Imagery::Blur(*eroded0, *blurred, 5);

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
	Imagery::Canny(*dilated, *canny, 50, 150);

	Square* bestSquare = new Square();
	Matrix* mainPixels = Imagery::ExtractBiggestPixelGroupAndCorners(*canny, 3, bestSquare);

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
		delete canny;
		delete mainPixels;
		delete[] lines;
		delete[] cartesianLines;

		return nullptr;
	}

	QPoint* vertices = new QPoint[] {
			(QPoint)bestSquare->topLeft,
			(QPoint)bestSquare->bottomRight,
			(QPoint)bestSquare->topRight,
			(QPoint)bestSquare->bottomLeft
	};

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

void DigitDetection(DetectionInfo* detectionInfo, const QString& savePath, int imgId, int* imgDigits)
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
	qDebug() << "9-removedLines";

	// Split the Sudoku into cells
	Matrix** cells = Imagery::Split(*perspective);
	Matrix** borderlessCells = Imagery::CropBorders((const Matrix**) cells, BORDER_CROP_PERCENTAGE);
	Matrix** resizedCells = Imagery::ResizeCellsTo28x28((const Matrix**) borderlessCells);
	bool* emptyCells = Imagery::GetEmptyCells((const Matrix**) resizedCells, EMPTY_CELL_THRESHOLD);
	Matrix** centeredCells = Imagery::CenterCells((const Matrix**) resizedCells, emptyCells);

	// Digit recognition
	Matrix* digits = new Matrix(9, 9, 1);
	NeuralNetwork* nn = NeuralNetwork::LoadFromFile("./nn.bin");
	for (int i = 0; i < 81; i++)
	{
		if (emptyCells[i])
		{
			digits->data[i] = 0;
			continue;
		}
		char* cellName = (char*) malloc(sizeof(char) * 40);
		sprintf(cellName, "digit_%i_%i_%i.png", imgDigits[i], imgId, i);
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
}


int countLines(const char* filename)
{
    FILE* file;
    int count = 0;
    char c;

    file = fopen(filename, "r"); // Open the file in read mode

    if (file == NULL)
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

// Todo: Finish auto rotation
// Todo: Link NN output with solver
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

        DetectionInfo * detection = BordersDetection(filePath, savePath);
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
