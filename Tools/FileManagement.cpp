//
// Created by mat on 30/12/23.
//

#include "FileManagement.h"
#include <QFileInfo>
#include <QDir>
#include <iostream>

namespace fs = std::filesystem;

namespace FileManagement
{
	QString GetFileName(const QString& path)
	{
		QFileInfo fileInfo(path);
		QString fileName(fileInfo.fileName()); // Get the file name with extension

		int lastDotIndex = (int) fileName.lastIndexOf('.');
		if (lastDotIndex != -1)
		{
			// If an extension exists, remove it
			fileName.truncate(lastDotIndex);
		}

		return fileName;
	}

// Removes all the files in a folder but not the folder itself
	void ClearFolder(const QString& path)
	{
		QDir directory(path);
		QStringList fileList = directory.entryList(QDir::NoDotAndDotDot | QDir::Files);

				foreach (QString fileName, fileList)
			{
				if (!directory.remove(fileName))
				{
					throw std::runtime_error("Error while clearing folder " + path.toStdString());
				}
			}
	}


	void CreateDirectory(const QString& path)
	{
		const std::string pathStr = path.toStdString();
		try
		{
			if (!fs::create_directory(pathStr))
			{
				if (errno != EEXIST)
				{
					throw std::system_error(errno, std::generic_category(),
											"Error while creating directory " + pathStr);
				}
			}
		} catch (const std::system_error& e)
		{
			std::cerr << e.what() << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
}
