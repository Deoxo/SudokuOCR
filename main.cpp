#include "mainwindow.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <filesystem>
#include <iostream>
#include <QFileInfo>
#include "Matrix.h"
#include "Tools/Settings.h"
#include "Imagery/Imagery.h"
#include "Imagery/GridDetection.h"
#include "NeuralNetwork/Network.h"
#include "Tools/FileManagement.h"

namespace fs = std::filesystem;

int GUI(int argc, char** argv)
{
    QApplication a(argc, argv);

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString& locale : uiLanguages)
    {
        const QString baseName = "SudokuOCR_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName))
        {
            a.installTranslator(&translator);
            break;
        }
    }
    MainWindow w;
    w.show();
    return a.exec();
}

typedef struct PreprocessInfo
{
    Matrix* dilated;
    Matrix* e;
} PreprocessInfo;

typedef struct DetectionInfo
{
    Square* bestSquare;
    List* squares;
    int numSquares;
    List* intersections;
    int numIntersections;
    Line* cartesianLines;
    int numLines;
} DetectionInfo;

PreprocessInfo* Preprocess(const QString& imgPath, const QString& savePath)
{
    // Convert the image to grayscale and stores it in a matrix
    Matrix* img = Imagery::LoadImageAsMatrix(imgPath);
    Matrix* m = Imagery::ConvertToGrayscale(*img);
    m->SaveAsImg(savePath, "0-grayscale.png");
    //printf("Standard deviation: %f\n", StandardDeviation(m, 5));

    // Noise reduction
    Matrix* bilaterallyFiltered = Matrix::CreateSameSize(*m);
    Imagery::BilateralFilter(*m, *bilaterallyFiltered, BIL_FIL_DIAM, BIL_FIL_SIG_COL, BIL_FIL_SIG_SPACE);
    bilaterallyFiltered->SaveAsImg(savePath, "1.0-bilaterallyFiltered.png");

    // Opening
    // Dilate
    Matrix* dilated0 = Matrix::CreateSameSize(*m);
    Imagery::Erode(*bilaterallyFiltered, *dilated0, DELATE_KERNEL_SIZE);
    dilated0->SaveAsImg(savePath, "1.1-dilated.png");

    // Erode
    Matrix* eroded0 = Matrix::CreateSameSize(*m);
    Imagery::Dilate(*dilated0, *eroded0, DELATE_KERNEL_SIZE);
    eroded0->SaveAsImg(savePath, "1.2-eroded.png");

    // Blur
    Matrix* blurred = Matrix::CreateSameSize(*m);
    Imagery::Blur(*eroded0, *blurred, 5);
    blurred->SaveAsImg(savePath, "1.3-blurred.png");

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
    binarized->SaveAsImg(savePath, "2-binarized.png");

    // Invert image
    Matrix* inversed = Matrix::CreateSameSize(*m);
    Imagery::BitwiseNot(*binarized, *inversed);
    inversed->SaveAsImg(savePath, "3-not.png");

    // Dilate
    Matrix* dilated = Matrix::CreateSameSize(*m);
    Imagery::Dilate(*inversed, *dilated, DELATE_KERNEL_SIZE);
    dilated->SaveAsImg(savePath, "4.0-dilated.png");

    Matrix* e = Matrix::CreateSameSize(*m);
    if (dispersion >= 500)
        Imagery::Erode(*dilated, *e, DELATE_KERNEL_SIZE);
    else
        dilated->CopyValuesTo(*e);
    e->SaveAsImg(savePath, "4.1-eroded.png");

    // Canny
    //Matrix* canny = MatCreateSameSize(dilated);
    //Canny(dilated, canny, 50, 150);

    // Get edges
    Matrix* sobelEdges = Matrix::CreateSameSize(*m);
    Imagery::SobelEdgeDetector(*dilated, *sobelEdges);
    sobelEdges->SaveAsImg(savePath, "5-sobelEdges.png");

    delete bilaterallyFiltered;
    delete dilated0;
    delete eroded0;
    delete blurred;
    delete binarized;
    delete inversed;
    delete sobelEdges;
    //MatFree(dilated);
    //MatFree(canny);

    PreprocessInfo* info = new PreprocessInfo();
    info->dilated = dilated;
    info->e = e;
    return info;
}

DetectionInfo* Detection(const PreprocessInfo* PreprocessInfo, const QString& savePath)
{
    Matrix* dilated = PreprocessInfo->dilated;
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
    int numIntersections = 0;
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

    Matrix* rotated = Imagery::Rotation(*PreprocessInfo->e, *bestSquare, -angle);
    rotated->SaveAsImg(savePath, "6-rotated.png");
    Matrix* cropped = Imagery::ExtractSudokuFromStraightImg(*rotated, *bestSquare, -angle);
    cropped->SaveAsImg(savePath, "7-cropped.png");
    Imagery::RemoveLines(*cropped);
    cropped->SaveAsImg(savePath, "8-removedLines.png");
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
        const QString cellName = "9-" + QString::number(i / 10) + QString::number(i % 10) + ".png";
        centeredCells[i]->SaveAsImg(savePath, cellName);

        *centeredCells[i] *= 1.f / 255.f;
        digits->data[i] = (float) nn->Predict(*centeredCells[i]) + 1;
    }

    digits->IntPrint();

    // Free memory
    //free(lines);
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
    delete digits;

    DetectionInfo* info = (DetectionInfo*) malloc(sizeof(DetectionInfo));
    info->bestSquare = bestSquare;
    info->squares = squares;
    info->numSquares = numSquares;
    info->intersections = intersections;
    info->numIntersections = numIntersections;
    info->cartesianLines = cartesianLines;
    info->numLines = numLines;

    return info;
}

int main(int argc, char** argv)
{
    QCoreApplication application(argc, argv);
    // Checks the number of arguments.
    if (argc != 3)
        throw std::runtime_error("Usage: image-file test/gui");

    // Sets selected mode
    const int gui = strcmp(argv[2], "test") == 0 ? 0 : (strcmp(argv[2], "gui") == 0 ? 1 : 2);

    // Quits if mode does not exist
    if (gui == 2)
        throw std::runtime_error("Usage: image-file test/gui");

    // Get the name of the image
    const QString* imgName = FileManagement::GetFileName(argv[1]);

    // Builds the folders to save the results
    QString savePath = QString("%1/%2/").arg(SAVE_FOLD, *imgName);
    FileManagement::CreateDirectory(SAVE_FOLD);
    FileManagement::CreateDirectory(savePath);
    FileManagement::ClearFolder(savePath);

    PreprocessInfo* preprocessInfo = Preprocess(argv[1], savePath);
    DetectionInfo* detectionInfo = Detection(preprocessInfo, savePath);

    int sw = preprocessInfo->e->cols, sh = preprocessInfo->e->rows;
    Square* bestSquare = detectionInfo->bestSquare;
    List* squares = detectionInfo->squares;
    int numSquares = detectionInfo->numSquares;
    List* intersections = detectionInfo->intersections;
    int numIntersections = detectionInfo->numIntersections;
    Line* cartesianLines = detectionInfo->cartesianLines;
    int numLines = detectionInfo->numLines;

    if (gui)
    {
        throw std::runtime_error("GUI not implemented yet");
    }

    // Free memory
    while (intersections != nullptr)
    {
        List* inter_li = (List*) intersections->data;
        ListDeepFree(inter_li);
        intersections = intersections->next;
    }
    ListFree(intersections);
    delete (cartesianLines);
    ListDeepFree(squares);
    delete preprocessInfo->dilated;
    delete preprocessInfo->e;
    delete preprocessInfo;
    delete detectionInfo;

    return EXIT_SUCCESS;
}
