#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "Tools/FileManagement.h"
#include "Tools/Settings.h"
#include <QImageReader>
#include <QMessageBox>
#include "GUI/ImgSelectorButton.h"
#include <QtConcurrent/QtConcurrent>
#include <QGraphicsPixmapItem>
#include "GUI/ShapeDefiner.h"
#include <QGraphicsEllipseItem>

MainWindow::MainWindow(QWidget* parent)
		: QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	ui->stackedWidget->setCurrentIndex(0);

	connect(this, &MainWindow::Quit, this, &MainWindow::close);
	connect(ui->ImageSelectorButton, &ImgSelectorButton::fileSelected, this, &MainWindow::OnImageSelected);
	connect(ui->backButton_1, &QPushButton::clicked, this, &MainWindow::GoToPage0);
	connect(ui->validateButton_1, &QPushButton::clicked, this, &MainWindow::OnImageValidated);
	connect(ui->validateButton_3, &QPushButton::clicked, this, &MainWindow::OnShapeValidated);

	core = new Core();
	ui->imgDisplay_2->setAlignment(Qt::AlignCenter);
	connect(core, &Core::StepCompleted, this, &MainWindow::OnStepCompleted);
	connect(core, &Core::OnVerticesDetected, this, &MainWindow::OnVerticesDetected);
	connect(core, &Core::OnDigitsRecognized, this, &MainWindow::OnDigitsRecognized);
}

MainWindow::~MainWindow()
{
	delete ui;
	delete core;
}

void MainWindow::OnImageSelected(const QString& filePath)
{
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
}

void MainWindow::OnImageValidated()
{
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
	ui->ShapeDefiner_3->SetImage(imgPath);
	ui->ShapeDefiner_3->SetVertices(vertices);
	ui->stackedWidget->setCurrentIndex(3);
}

void MainWindow::OnShapeValidated()
{
	ui->imgDisplay_4->SetImage(imgPath);
	ui->stackedWidget->setCurrentIndex(4);
	core->DigitDetection(detectionInfo, savePath);
}

void MainWindow::OnDigitsRecognized(const Matrix* digits)
{
	digits->IntPrint();
	QGridLayout * gridLayout = ui->gridLayout_4_;

	const int rowCount = gridLayout->rowCount();
	const int columnCount = gridLayout->columnCount();

	for (int row = 0; row < rowCount; ++row)
	{
		for (int col = 0; col < columnCount; ++col)
		{
			// Get the item at the current row and column
			QLayoutItem* item = gridLayout->itemAtPosition(row, col);


			if (item)
			{
				// Handle the widget or layout item
				QWidget * widget = item->widget();
				if (widget)
				{
					QSpinBox* spinBox = qobject_cast<QSpinBox*>(widget);
					if (spinBox)
						spinBox->setValue((int) digits->data[(row - 1) * columnCount + col]);

				}
			}
		}
	}
	ui->validateButton_4->setEnabled(true);
}
