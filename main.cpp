#include "mainwindow.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <QFileInfo>
#include "Imagery/Imagery.h"
#include "Tools/FileManagement.h"

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
	// Checks the number of arguments.
	if (argc != 3)
		throw std::runtime_error("Usage: image-file test/gui");

	// Sets selected mode
	const int gui = strcmp(argv[2], "test") == 0 ? 0 : (strcmp(argv[2], "gui") == 0 ? 1 : 2);

	if (gui)
	{
		char** v = new char* [1];
		v[0] = argv[0];
		int r = GUI(1, argv);
		delete[] v;
		return r;
	}

	// Quits if mode does not exist
	if (gui == 2)
		throw std::runtime_error("Usage: image-file test/gui");

	// Get the name of the image
	const QString* imgName = FileManagement::GetFileName(argv[1]);

	Imagery::ProcessImage(*imgName);
	return 0;
}
