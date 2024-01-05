#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QImageReader>
#include <QFileInfo>
#include <QMessageBox>
#include <QTimer>
#include <ImgSelectorButton.h>

MainWindow::MainWindow(QWidget* parent)
		: QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	ui->stackedWidget->setCurrentIndex(0);

	connect(this, &MainWindow::Quit, this, &MainWindow::close);
	connect(ui->ImageSelectorButton, &ImgSelectorButton::fileSelected, this, &MainWindow::OnImageSelected);
	connect(ui->backButton_1, &QPushButton::clicked, this, &MainWindow::GoToPage0);
	connect(ui->validateButton_1, &QPushButton::clicked, this, &MainWindow::OnImageValidated);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::OnImageSelected(const QString& filePath)
{
	imgPath = QString(filePath);
	// Switch to page 2
	ui->stackedWidget->setCurrentIndex(1);
	ui->imgDisplay_1->SetImage(filePath);
	//Imagery::ProcessImage(filePath);
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
}
