#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCloseEvent>
#include "Core.h"

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

signals:

	void Quit();

public slots:

	void OnImageSelected(const QString& filePath);

	void GoToPage0();

	void OnImageValidated();

	void OnStepCompleted(const QString& stepName);

private:
	Ui::MainWindow* ui;
	QString imgPath;
	QString savePath;
	Core* core;
};

#endif // MAINWINDOW_H
