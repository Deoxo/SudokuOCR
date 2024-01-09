//
// Created by mat on 30/12/23.
//

#ifndef SUDOKUOCR_FILEMANAGEMENT_H
#define SUDOKUOCR_FILEMANAGEMENT_H

#include <QString>

namespace FileManagement
{
	// Returns the name of the file at the end of the path without the extension
	QString GetFileName(const QString& path);

	// Removes all the files in a folder but not the folder itself
	void ClearFolder(const QString& path);

	void CreateDirectory(const QString& path);
}

#endif //SUDOKUOCR_FILEMANAGEMENT_H
