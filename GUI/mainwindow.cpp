#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "Tools/FileManagement.h"
#include "Tools/Settings.h"
#include <QImageReader>
#include <QMessageBox>
#include "GUI/ImgSelectorButton.h"
#include <QtConcurrent/QtConcurrent>
#include "Tools/Solver.h"
#include <QGraphicsEllipseItem>
#include <iostream>

MainWindow::MainWindow(QWidget* parent)
		: QMainWindow(parent), ui(new Ui::MainWindow)
{
	setWindowTitle("SudokuOCR - Menu");
	ui->setupUi(this);
	ui->stackedWidget->setCurrentIndex(0);

	connect(this, &MainWindow::Quit, this, &MainWindow::close);
	connect(ui->ImageSelectorButton, &ImgSelectorButton::fileSelected, this, &MainWindow::OnImageSelected);
	connect(ui->backButton_1, &QPushButton::clicked, this, &MainWindow::GoToPage0);
	connect(ui->validateButton_1, &QPushButton::clicked, this, &MainWindow::OnImageValidated);
	connect(ui->validateButton_3, &QPushButton::clicked, this, &MainWindow::OnShapeValidated);
	connect(ui->validateButton_4, &QPushButton::clicked, this, &MainWindow::OnAllDigitsValidated);
	connect(ui->menuButton_5, &QPushButton::clicked, this, &MainWindow::GoToPage0);

	core = new Core();
	ui->imgDisplay_2->setAlignment(Qt::AlignCenter);
	connect(core, &Core::StepCompleted, this, &MainWindow::OnStepCompleted);
	connect(core, &Core::OnVerticesDetected, this, &MainWindow::OnVerticesDetected);
	connect(core, &Core::OnDigitsRecognized, this, &MainWindow::OnDigitsRecognized);

	QGridLayout * gridLayout = ui->gridLayout_4_;
	for (int row = 1; row < 10; ++row)
	{
		for (int col = 0; col < 9; ++col)
		{
			// Get the item at the current row and column
			int r = row, c = col;
			if (r >= 4 && r <= 6)
				r++;
			else if (r >= 7)
				r += 2;
			if (c >= 3 && c <= 5)
				c++;
			else if (c >= 6)
				c += 2;
			QLayoutItem* item = gridLayout->itemAtPosition(r, c);


			if (item)
			{
				// Handle the widget or layout item
				QWidget * widget = item->widget();
				if (widget)
				{
					QSpinBox* spinBox = qobject_cast<QSpinBox*>(widget);
					if (spinBox)
						spinBoxes[(row - 1) * 9 + col] = spinBox;
				}
			}
		}
	}
}

MainWindow::~MainWindow()
{
	delete ui;
	delete core;
}

void MainWindow::OnImageSelected(const QString& filePath)
{
	setWindowTitle("SudokuOCR - Validation");
	imgPath = QString(filePath);
	// Switch to page 2
	ui->stackedWidget->setCurrentIndex(1);
	ui->imgDisplay_1->SetImage(filePath);

	// Builds the folders to save the results
	const QString imageName = FileManagement::GetFileName(filePath);
	savePath = QDir::currentPath() + QString("/%1/%2/").arg(SAVE_FOLD, imageName);
	FileManagement::CreateDirectory(SAVE_FOLD);
	FileManagement::CreateDirectory(savePath);
	FileManagement::ClearFolder(savePath);
	// Schedule closing of the window after processing the current events
	//QTimer::singleShot(1000, this, &MainWindow::close);
}

void MainWindow::GoToPage0()
{
	ui->stackedWidget->setCurrentIndex(0);
	setWindowTitle("SudokuOCR - Menu");
}

void MainWindow::OnImageValidated()
{
	setWindowTitle("SudokuOCR - Preprocessing");
	ui->stackedWidget->setCurrentIndex(2);
	ui->imgDisplay_2->SetImage(imgPath);
	connect(&watcher, &QFutureWatcher<DetectionInfo*>::finished, this, [&]()
	{ detectionInfo = watcher.result(); });
	QFuture<DetectionInfo*> future = QtConcurrent::run([this]
													   { return core->BordersDetection(imgPath, savePath); });
	watcher.setFuture(future);
}

void MainWindow::OnStepCompleted(const QString& stepName)
{
	qDebug() << stepName;
	ui->imgDisplay_2->SetImage(savePath + stepName + ".png");
}

void MainWindow::OnVerticesDetected(QPoint* vertices)
{
	setWindowTitle("SudokuOCR - Borders detection");
	ui->ShapeDefiner_3->SetImage(imgPath);
	ui->ShapeDefiner_3->SetVertices(vertices);
	ui->stackedWidget->setCurrentIndex(3);
}

void MainWindow::OnShapeValidated()
{
	setWindowTitle("SudokuOCR - Digits detection");
	ui->imgDisplay_4->SetImage(imgPath);
	ui->stackedWidget->setCurrentIndex(4);
	core->DigitDetection(detectionInfo, savePath);
}

void MainWindow::OnDigitsRecognized(const Matrix* digits)
{
	digits->IntPrint();
	for (int i = 0; i < 81; ++i)
	{
		spinBoxes[i]->setValue((int) digits->data[i]);
		connect(spinBoxes[i], &QSpinBox::valueChanged, this, &MainWindow::OnDigitModified);
	}

	ui->validateButton_4->setEnabled(true);
	OnDigitModified();
}

void MainWindow::OnDigitModified()
{
	Matrix* board = new Matrix(9, 9, 1);
	for (int i = 0; i < 81; ++i)
		board->data[i] = (float) spinBoxes[i]->value();

	Matrix* result = Solver::Solve(*board);
	if (result != nullptr)
	{
		ui->validateButton_4->setEnabled(true);
		delete result;
	}
	else ui->validateButton_4->setEnabled(false);
}

void MainWindow::OnAllDigitsValidated()
{
	setWindowTitle("SudokuOCR - Result");
	Matrix* board = new Matrix(9, 9, 1);
	for (int i = 0; i < 81; ++i)
		board->data[i] = (float) spinBoxes[i]->value();

	Matrix* result = Solver::Solve(*board);
	core->SaveSudokuImage(*result, 600, savePath + "result.png");
	ui->imgDisplay_5->SetImage(savePath + "result.png");
	ui->stackedWidget->setCurrentIndex(5);
}
