#include "GUI/mainwindow.h"

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
			QApplication::installTranslator(&translator);
			break;
		}
	}
	MainWindow w;
	w.show();
	return QApplication::exec();
}

// Todo: Once perspective has been done, check for white borders and crop them (at least do not take them for the split)
// Todo: Check for need to crop cells borders (for example by computing the ratio of white pixels along each row and column)
int main(int argc, char** argv)
{
	return GUI(argc, argv);
}
