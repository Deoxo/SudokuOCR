#include "mainwindow.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <QFileInfo>

namespace fs = std::filesystem;

int GUI(int argc, char** argv)
{
	QApplication a(argc, argv);
	QApplication::setQuitOnLastWindowClosed(true);

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

int main(int argc, char** argv)
{
	return GUI(argc, argv);
}
