#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCloseEvent>
#include "Core.h"
#include <QGraphicsScene>
#include <QFutureWatcher>
#include <QSpinBox>

QT_BEGIN_NAMESPACE
namespace Ui
{
	class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
Q_OBJECT

public:
	MainWindow(QWidget* parent = nullptr);

	~MainWindow();

	void OnDigitsRecognized(const Matrix* digits);

signals:

	void Quit();

public slots:

	void OnImageSelected(const QString& filePath);

	void GoToPage0();

	void OnImageValidated();

	void OnStepCompleted(const QString& stepName);

	void OnVerticesDetected(QPoint* vertices);

	void OnShapeValidated();

	void OnDigitModified();

private:
	Ui::MainWindow* ui;
	QString imgPath;
	QString savePath;
	Core* core;

	DetectionInfo* detectionInfo;
	QFutureWatcher<DetectionInfo*> watcher;

	QSpinBox* spinBoxes[81];
};

#endif // MAINWINDOW_H
