#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "Tools/FileManagement.h"
#include "Tools/Settings.h"
#include <QImageReader>
#include <QMessageBox>
#include <ImgSelectorButton.h>
#include <QtConcurrent/QtConcurrent>

MainWindow::MainWindow(QWidget* parent)
		: QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	ui->stackedWidget->setCurrentIndex(0);

	connect(this, &MainWindow::Quit, this, &MainWindow::close);
	connect(ui->ImageSelectorButton, &ImgSelectorButton::fileSelected, this, &MainWindow::OnImageSelected);
	connect(ui->backButton_1, &QPushButton::clicked, this, &MainWindow::GoToPage0);
	connect(ui->validateButton_1, &QPushButton::clicked, this, &MainWindow::OnImageValidated);

	core = new Core();
	connect(core, &Core::StepCompleted, this, &MainWindow::OnStepCompleted);
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
	//core->ProcessImage(imgPath, savePath);
	QFuture<void> _ = QtConcurrent::run([=]()
										{ core->ProcessImage(imgPath, savePath); });
}

void MainWindow::OnStepCompleted(const QString& stepName)
{
	qDebug() << stepName;
	ui->imgDisplay_2->SetImage(savePath + stepName + ".png");
}
